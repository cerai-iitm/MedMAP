# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
import datasets
from datasets import Dataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    TrainerCallback, TrainerState, TrainerControl, logging,
    DataCollatorForLanguageModeling, PreTrainedTokenizer, training_args, modeling_utils
)
from dp_transformers.custom_trainer import Trainer
from transformers.file_utils import is_sagemaker_mp_enabled, is_datasets_available
import opacus
from opacus.accountants import RDPAccountant
# MedMAP modification: the prv_accountant import was removed. All of its usages
# were already commented out in this fork (privacy accounting is done with
# Google dp_accounting in arguments.find_noise_multiplier), so the import only
# forced an unnecessary dependency.
# from prv_accountant import Accountant as PRVAccountant
from contextlib import contextmanager
from typing import Any, Callable, List, Optional, Union, Dict, Sequence
from accelerate.optimizer import AcceleratedOptimizer

from dp_transformers import sampler, arguments, custom_dp_optimizer, custom_ddp_optimizer, custom_adam
from torch.utils.data import default_collate


logger = logging.get_logger(__name__)

def print_gpu_memory(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"{prefix}Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")


class DPCallback(TrainerCallback):
    """
    This class registers all the necessary callbacks to make transformers.Trainer compatible with opacus.
    """
    def __init__(
        self,
        noise_multiplier: float,
        target_delta: float,
        sampling_probability: float,
        # rdp_accountant: RDPAccountant,
        # prv_accountant: PRVAccountant,
        max_epsilon: float = float('inf')
    ) -> None:

        self.noise_multiplier = noise_multiplier
        self.target_delta = target_delta
        self.sampling_probability = sampling_probability
        # self.rdp_accountant = rdp_accountant
        # self.prv_accountant = prv_accountant

        self.max_epsilon = max_epsilon
        self.on_substep_end_was_called = False
        # self.compute_rdp_epsilon = lambda: self.rdp_accountant.get_epsilon(self.target_delta)
        # self.compute_prv_epsilon = lambda s: self.prv_accountant.compute_epsilon(s)[2]

    def on_substep_end(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, optimizer=None, **kwargs):
        if optimizer is None:
            raise RuntimeError("Impossible to access optimizer from inside callback")
        if isinstance(optimizer, AcceleratedOptimizer):
            dp_optimizer = optimizer.optimizer
        else:
            dp_optimizer = optimizer
        
        dp_optimizer.step()
        dp_optimizer.zero_grad()

        self.on_substep_end_was_called = True

    def on_step_end(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, optimizer=None, **kwargs):
        if not (
            args.gradient_accumulation_steps <= 1 or
            self.on_substep_end_was_called
        ):
            raise RuntimeError(
                "Gradient accumulation was specified but `on_substep_end` wasn't called. "
                "Make sure you're using a recent version of transformers (>=4.10.0) "
                "which has an appropriate callback in the trainer."
            )

        if optimizer is None:
            raise RuntimeError("Impossible to access optimizer from inside callback")
        optimizer.zero_grad()  # Opacus is bothered that HF does not call .zero_grad() on the optimizer

        # self.rdp_accountant.step(noise_multiplier=self.noise_multiplier, sample_rate=self.sampling_probability)

    # def on_save(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     return self._check_max_epsilon_exceeded(state, control)

    # def on_evaluate(self, args: training_args.TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     return self._check_max_epsilon_exceeded(state, control)

    # def _check_max_epsilon_exceeded(self, state: TrainerState, control: TrainerControl) -> TrainerControl:
    #     eps_rdp = self.compute_rdp_epsilon()
    #     eps_prv = self.compute_prv_epsilon(state.global_step)
    #     if eps_rdp > self.max_epsilon or eps_prv > self.max_epsilon:
    #         logger.error("Max epsilon exceeded. Stopping training...")
    #         control.should_training_stop = True
    #     return control


class DataCollatorForPrivateCausalLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer=tokenizer, mlm=False)

    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(examples)

        # Huggingface's default way of constructing position_ids is not compatible with Opacus
        # since Opacus is not able to deduce the batch size from the input. Here we manually
        # generate a position_ids tensor which has the same values as Huggingface's default tensor
        # but it is constructed in a way that is compatile with Opacus by using expand_as.
        if "position_ids" not in batch:
            input_ids = batch["input_ids"]
            batch["position_ids"] = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).repeat(input_ids.shape[0], 1)
        return batch


class GradSampleModule(opacus.GradSampleModule):
    """
    Little wrapper to provide `no_sync` context which is assumed by Huggingface trainer.
    We don't need to do anything in addition here
    """
    @contextmanager
    def no_sync(self):
        yield


def create_author_mapping(dataset: Dataset, author: str) -> Sequence[Sequence[int]]:
    """
    Creates a mapping from authors to samples in a dataset.
    """
    with dataset.formatted_as(type="pandas"):
        authors = pd.DataFrame(data={"author": dataset[author]})
        author_mapping = [g.index.values for _, g in authors.groupby("author")]
    return author_mapping


class OpacusDPTrainer(Trainer):
    """
    Wrapper to modify Huggingface Trainer to:
        (i) remove "loss = loss / self.args.gradient_accumulation_steps" operation in training_step
        as this is already handled by Opacus package.
        (ii) enable author-level DP training by modifing the sampler and the dataloader. In the case
        of sample-level DP, each sample can be represented by a unique author.
        (iii) wrap the optimizer with Opacus' DPOptimizer/DistributedDPOptimizer
    """
    def __init__(
        self,
        model: Union[modeling_utils.PreTrainedModel, torch.nn.modules.module.Module] = None,
        args: arguments.TrainingArguments = None,
        train_dataset: Optional[torch.utils.data.dataset.Dataset] = None,
        privacy_args: arguments.PrivacyArguments = None,
        author_mapping: Optional[Sequence[Sequence[int]]] = None,
        **kwargs: Dict
    ) -> None:

        self.train_args = args
        self.user_grad_acc = self.train_args.gradient_accumulation_steps
        assert (self.train_args.per_device_train_batch_size %  self.user_grad_acc == 0), "grad_accum not divisible by batch_size"
        self.privacy_args = privacy_args

        # Sample-level DP is equivalent to mapping each sample to a unique author. 
        if author_mapping is None:
            author_mapping = [[i] for i in range(len(train_dataset))]
        self.author_mapping = author_mapping

        if not self.privacy_args.is_initialized:
            self.privacy_args.initialize(
                sampling_probability=self.sampling_probability,
                num_steps=self.num_steps,
                num_samples=len(self.author_mapping),
            )

        # Wrap model in DDP and GradSampleModule
        if args.parallel_mode == training_args.ParallelMode.DISTRIBUTED:
            print("Wrapping Model with DPDDP training")
            logger.info(f"Wrapping the model with DPDDP in distributed training.")
            model = opacus.distributed.DifferentiallyPrivateDistributedDataParallel(model)
        
        model = GradSampleModule(model)  #makes sure model.parameters has the grad_sample attribute

        # Set up callback for accounting and handling grad acc
        self.dp_callback = DPCallback(
            noise_multiplier=self.privacy_args.noise_multiplier,
            target_delta=self.privacy_args.target_delta,
            sampling_probability=self.sampling_probability,
            # rdp_accountant=self.rdp_accountant,
            # prv_accountant=self.prv_accountant    
        )
        callbacks_list = kwargs.pop("callbacks", [])
        callbacks_list = [self.dp_callback, *callbacks_list]
        args.gradient_accumulation_steps = 1
        super().__init__(model=model, args=args, train_dataset=train_dataset, callbacks=callbacks_list, public_batch=self.privacy_args.feature_dp, **kwargs)
        self._accumulated_grad_samples = None  # will be list of tensors per param or None

    @property
    def sampling_probability(self) -> float:
        return self.train_args.per_device_train_batch_size * self.train_args.world_size  / len(self.author_mapping)

    @property
    def num_steps(self) -> int:
        return int(self.train_args.num_train_epochs * (1 / self.sampling_probability))

    def create_optimizer(self):
        _ = super().create_optimizer()

        if self.args.parallel_mode == training_args.ParallelMode.DISTRIBUTED:
            optimizer_generator = custom_ddp_optimizer.DistributedDPOptimizer

        else:
            optimizer_generator = custom_dp_optimizer.DPOptimizer

        # removed loss_reduction feature, assumes gradients provided is averaged acc to overall batch_size per device 
        self.optimizer = optimizer_generator(
            optimizer=self.optimizer,
            noise_multiplier=self.privacy_args.noise_multiplier,
            max_grad_norm=self.privacy_args.per_sample_max_grad_norm,
            expected_batch_size=self.args.per_device_train_batch_size,
            feature_dp=self.privacy_args.feature_dp,
            gradient_accumulation_steps=self.train_args.gradient_accumulation_steps, #still actual gradient_acc steps
        )

        print("Created Optimizer with Sigma:", self.optimizer.noise_multiplier)
        return self.optimizer
    
    def clip(self, grad_sample, max_norm, batch_size):
        # filter out None
        valid_grads = [g for g in grad_sample if g is not None]
        if len(valid_grads) == 0:
            return grad_sample

        per_param_norms = []

        for g in grad_sample:
            if g is None:
                continue
            else:
                # already per-sample
                per_param_norms.append(g.view(batch_size, -1).norm(2, dim=1) ** 2)

        # aggregate norm
        total_norm = torch.stack(per_param_norms).sum(dim=0).sqrt()  # [B]
        factors = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)

        # now clip
        clipped = []
        for g in grad_sample:
            if g is None:
                clipped.append(None)
                continue
            else:
                clipped.append(g * factors.view(-1, *([1] * (g.dim() - 1))))

        return clipped

    
    def _split_batch(self, inputs, num_microbatches: int):
        """
        Splits a batch (dict of tensors or tensor) into `num_microbatches`
        along the first (batch) dimension.

        Args:
            inputs: batch from dataloader (can be a dict, tuple, or tensor).
            num_microbatches: number of microbatches to create.

        Returns:
            A list of microbatches, each shaped consistently.
        """

        if num_microbatches == 1:
            return [inputs]

        print(type(inputs))
        
        # get batch size
        if isinstance(inputs, dict):
            batch_size = next(iter(inputs.values())).size(0)
        elif isinstance(inputs, (list, tuple)):
            batch_size = inputs[0].size(0)
        else:  # Tensor
            batch_size = inputs.size(0)

        microbatch_size = (batch_size + num_microbatches - 1) // num_microbatches
        microbatches = []

        for start in range(0, batch_size, microbatch_size):
            end = min(start + microbatch_size, batch_size)
            if isinstance(inputs, dict):
                mb = {k: v[start:end] for k, v in inputs.items()}
            elif isinstance(inputs, (list, tuple)):
                mb = [x[start:end] for x in inputs]
            else:
                mb = inputs[start:end]
            microbatches.append(mb)

        return microbatches
    
    def _init_accumulation_buffers(self, model):
        """Create empty buffers matching the model.parameters() structure."""
        params = list(model.parameters())
        self._accumulated_grad_samples = [None for _ in params]

    def _clear_accumulation_buffers(self):
        self._accumulated_grad_samples = None
    
    def _accumulate_grad_samples(self, model):
        """
        Instead of concatenating per-sample gradients across microbatches,
        we simply sum them into the accumulation buffer. At the end, we divide
        by the number of microbatches (user_grad_accum) to get the average.
        """
        for i, p in enumerate(model.parameters()):
            if not hasattr(p, "grad_sample") or p.grad_sample is None:
                continue

            if self._accumulated_grad_samples[i] is None:
                # Initialize buffer with the first microbatch
                self._accumulated_grad_samples[i] = p.grad_sample.detach().clone()
            else:
                # Sum the gradients across microbatches
                self._accumulated_grad_samples[i] += p.grad_sample.detach()

            # Clear grad_sample to avoid double counting in next microbatch
            p.grad_sample = None

        
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], public_inputs: Optional[Dict[str, Union[torch.Tensor, Any]]], num_items_in_batch=None) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        microbatches = self._split_batch(inputs, self.user_grad_acc)
        if self.public_batch:
            public_microbatches = self._split_batch(public_inputs, self.user_grad_acc)
        else:
            public_microbatches = [None] * len(microbatches)

        self._init_accumulation_buffers(model)

        if isinstance(self.optimizer, AcceleratedOptimizer):
            original_optimizer = self.optimizer.optimizer.original_optimizer
        else:
            original_optimizer = self.optimizer.original_optimizer

        if isinstance(original_optimizer, custom_adam.CustomAdamW):
            self.public_buffer = [None for _ in self._accumulated_grad_samples]

        total_loss = 0.0
        for inputs, public_inputs in zip(microbatches, public_microbatches):
            inputs = self._prepare_inputs(inputs)
            if self.public_batch:
                public_inputs = self._prepare_inputs(public_inputs)
            microbatch_size = inputs["labels"].shape[0]

            if is_sagemaker_mp_enabled():
                raise NotImplementedError("DP currently doesn't support this")

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, public=False, feature_dp=self.privacy_args.feature_dp)
                # print("Computed Net Loss")
                # print_gpu_memory()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average as reduction is none

            #------------------------------------------------------------------------
            if self.use_apex:
                raise NotImplementedError("DP currently doesn't support this")
            else:
                # print("Calculating Net Gradients...")
                loss.backward()
                # print_gpu_memory()
            #-------------------------------------------------------------------------

            if self.privacy_args.feature_dp:  
                priv_grads = [
                    p.grad_sample.clone() / (self.user_grad_acc) if hasattr(p, "grad_sample") and p.grad_sample is not None else None
                    for p in model.parameters()
                ]
                # print("Stored Priv Grads")

                self.optimizer.zero_grad(set_to_none=True)

                ## PUBLIC GRADIENT CALCULATION:
                with self.compute_loss_context_manager():
                    pub_loss = self.compute_loss(model, public_inputs, public=True, feature_dp=True)
                    # print("Computed Public Loss")
                    # print_gpu_memory()


                if self.args.n_gpu > 1:
                    pub_loss = pub_loss.mean()  # mean() to average on multi-gpu parallel training

                if self.use_apex:
                    raise NotImplementedError("DP currently doesn't support this")
                else:
                    # print("Calculating Public Gradients...")
                    pub_loss.backward()
                    # print_gpu_memory()
 
                pub_grads = [
                    p.grad_sample.clone() / (self.user_grad_acc) if hasattr(p, "grad_sample") and p.grad_sample is not None else None
                    for p in model.parameters()
                ]

                pub_grads = self.clip(pub_grads, self.privacy_args.public_clip, microbatch_size)
                
                priv_grads = self.clip(priv_grads, self.privacy_args.per_sample_max_grad_norm, microbatch_size)
                
                if isinstance(original_optimizer, custom_adam.CustomAdamW):
                    # print("Buffering Public Gradients separately")
                    assert len(pub_grads) == len(self.public_buffer), "length of private buffer and private grads unequal"
                    for i in range(len(self.public_buffer)):
                        if pub_grads[i] is None:
                            pass
                        elif self.public_buffer[i] is None:
                            self.public_buffer[i] = pub_grads[i]
                        else:
                            self.public_buffer[i] += pub_grads[i]
                            
                grads = []
                for a,b in zip(pub_grads, priv_grads):
                    if a is None and b is None:
                        grads.append(None)
                    elif a is None:
                        grads.append(b)
                    elif b is None:
                        grads.append(a)
                    else:
                        grads.append(a+b)

                for p, g in zip(model.parameters(), grads):
                    if hasattr(p, "grad_sample"):
                        p.grad_sample = g
                    elif g is not None:
                        raise ValueError("Gradients found for a frozen parameter")

                # print("Merged Gradients...")

            else:
                net_grads = [
                        p.grad_sample.clone() / (self.user_grad_acc) if hasattr(p, "grad_sample") and p.grad_sample is not None else None
                        for p in model.parameters()
                    ]
                net_grads = self.clip(net_grads, self.privacy_args.per_sample_max_grad_norm, microbatch_size)
                # print("Clipped Net Grads")
                for p, g in zip(model.parameters(), net_grads):
                    if hasattr(p, "grad_sample"):
                        p.grad_sample = g
                    elif g is not None:
                        raise ValueError("Gradients found for a frozen parameter")
            
            self._accumulate_grad_samples(model)
            total_loss += loss.detach()
        
        if self.privacy_args.feature_dp and isinstance(original_optimizer, custom_adam.CustomAdamW):
            print("Public step with second moment update")
            for p, pub in zip(model.parameters(), self.public_buffer):
                if pub is not None:
                    p.grad = pub.sum(dim=0).to(p.dtype)
            
            original_optimizer.step(update_first_moment=False, update_second_moment=True, increment_step=True)
            self.optimizer.zero_grad(set_to_none=True)
            self.public_buffer = None

        for p, summed in zip(model.parameters(), self._accumulated_grad_samples):
            if summed is not None:
                # No need to divide by num of microbatches, already done while storing
                p.grad_sample = summed
        
        self._clear_accumulation_buffers()
                
        return total_loss/self.user_grad_acc

    def _get_train_sampler(self):
        """
        Provides author sampler.
        """
        train_sampler = sampler.ShuffledAuthorSampler(
            author_mapping=self.author_mapping,
            batch_size=self.args.per_device_train_batch_size,
            world_size=self.args.world_size
        )
        return train_sampler

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use the author-level sampler from dp_transformers.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = self._get_train_sampler()

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        return DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )