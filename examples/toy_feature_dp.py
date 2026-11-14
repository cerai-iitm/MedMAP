#!/usr/bin/env python3
"""Introductory example: a two-modality MNIST task under the MedMAP trainer stack.

This is the smallest end-to-end illustration of what MedMAP standardizes. It
needs no credentialed data and runs on a CPU in a few minutes.

Task
----
- Public modality  x_pub : the MNIST digit image, average-pooled to a coarse
                           --public-resolution x --public-resolution grid (7x7 =
                           49 pixels by default), treated like the imaging
                           modality of a MedMAP dataset. The pooling makes the
                           image informative but imperfect, so the private
                           modality has something to add (the public features of
                           Mahloujifar et al., 2025 were blurred images too).
- Private modality x_priv: a 10-dimensional "clinical hint", a one-hot
                           encoding of the digit that is wrong with probability
                           --hint-noise. It is informative but, by construction,
                           the thing we want to protect (like a clinical report
                           or questionnaire).
- Label y                : the digit (10 classes).

Four arms are trained on identical data, exactly as in the paper:

    image-only      non-private model that never sees x_priv (privacy-safe floor)
    both            non-private model on (x_pub, x_priv)   (utility ceiling)
    dp-sgd          every example's whole gradient is clipped and noised
    feature-dp      only the x_priv gradient residual is clipped and noised

Usage
-----
    python -m examples.toy_feature_dp                      # downloads MNIST
    python -m examples.toy_feature_dp --synthetic          # no download
    python -m examples.toy_feature_dp --epsilon 1 --arms both feature-dp

What to look for: the hint lifts accuracy well above the image-only floor,
Feature-DP keeps part of that lift at small epsilon, and DP-SGD, which also
pays to protect the pixels, retains less (how much depends on epsilon, the
learning rate and the clip norms, exactly the trade-offs MedMAP measures).

The example exercises the same three pieces that a new dataset or model needs
(see README, "Extending MedMAP"): a model that honours the
``public= / feature_dp= / loss_mode=`` forward contract, a collator that
returns a dict of tensors with a ``labels`` key, and ``get_trainer``.
"""

from __future__ import annotations

import os
import sys

# Allow both `python -m examples.toy_feature_dp` (from the repo root) and
# `python examples/toy_feature_dp.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._bootstrap import setup_project_paths  # noqa: E402

setup_project_paths()

import argparse
import tempfile
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import TrainingArguments

from dp_transformers.arguments import PrivacyArguments
from medmap.training.trainers import get_trainer


# --------------------------------------------------------------------------- #
# 1. Data: public image + private hint
# --------------------------------------------------------------------------- #
class TwoModalityDataset(Dataset):
    """Each item is a dict with public_features, private_features, labels."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, hint_noise: float, seed: int,
                 public_resolution: int = 7):
        rng = np.random.default_rng(seed)
        n = len(labels)
        hinted = labels.copy()
        flip = rng.random(n) < hint_noise
        hinted[flip] = rng.integers(0, 10, size=int(flip.sum()))
        img = torch.tensor(images.reshape(n, 1, 28, 28) / 255.0, dtype=torch.float32)
        img = F.adaptive_avg_pool2d(img, public_resolution)      # coarse "imaging" modality
        self.public = img.reshape(n, -1)
        self.private = F.one_hot(torch.tensor(hinted, dtype=torch.long), 10).float()
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"public_features": self.public[i], "private_features": self.private[i], "labels": self.labels[i]}


def load_mnist(root: str, n_train: int, n_test: int, seed: int):
    from torchvision.datasets import MNIST

    tr, te = MNIST(root, train=True, download=True), MNIST(root, train=False, download=True)
    rng = np.random.default_rng(seed)
    i_tr = rng.choice(len(tr), n_train, replace=False)
    i_te = rng.choice(len(te), n_test, replace=False)
    return (tr.data.numpy()[i_tr], tr.targets.numpy()[i_tr]), (te.data.numpy()[i_te], te.targets.numpy()[i_te])


def load_synthetic(n_train: int, n_test: int, seed: int):
    """Ten Gaussian blobs in 784-d that stand in for MNIST when offline."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, size=(10, 784))

    def make(n):
        y = rng.integers(0, 10, size=n)
        x = centers[y] + rng.normal(0, 3.0, size=(n, 784))
        x = np.clip((x - x.min()) / (x.max() - x.min()) * 255, 0, 255)
        return x.astype(np.float32), y

    return make(n_train), make(n_test)


def collate(features):
    return {k: torch.stack([f[k] for f in features]) for k in features[0]}


# --------------------------------------------------------------------------- #
# 2. Model: honours the MedMAP forward contract
# --------------------------------------------------------------------------- #
class TwoModalityMLP(nn.Module):
    """MLP on [x_pub, x_priv].

    loss_mode="standard":   loss = CE(model(x_pub, x_priv), y)
    loss_mode="decomposed": public=True        -> loss = CE(model(x_pub, 0), y)
                            feature_dp=True    -> loss = CE(model(x_pub, x_priv), y) - CE(model(x_pub, 0), y)
                            feature_dp=False   -> loss = CE(model(x_pub, x_priv), y)
    The DP trainer differentiates these per example, so the Feature-DP branch
    yields exactly the gradient attributable to x_priv.
    """

    def __init__(self, public_dim: int, private_dim: int = 10, hidden: int = 256, num_classes: int = 10,
                 use_private: bool = True):
        super().__init__()
        self.use_private = use_private
        self.net = nn.Sequential(nn.Linear(public_dim + private_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))

    def forward(self, public_features, private_features, labels=None, public: bool = False,
                feature_dp: bool = False, loss_mode: str = "standard", **kwargs) -> Dict[str, torch.Tensor]:
        if not self.use_private:
            private_features = torch.zeros_like(private_features)
        x_full = torch.cat([public_features, private_features], dim=-1)
        x_pub = torch.cat([public_features, torch.zeros_like(private_features)], dim=-1)
        if self.training and loss_mode == "decomposed":
            # Opacus needs the first layer's input to require grad to record per-sample gradients.
            x_full, x_pub = x_full.detach().requires_grad_(True), x_pub.detach().requires_grad_(True)

        def ce(logits):
            return F.cross_entropy(logits, labels) if labels is not None else None

        # Run only the forwards that feed the loss: an unused extra forward through
        # the same module breaks Opacus's per-sample gradient bookkeeping.
        if loss_mode == "standard" or not (public or feature_dp):
            logits = self.net(x_full)                      # non-private, or full DP-SGD
            loss = ce(logits)
        elif public:
            logits = self.net(x_pub)                       # public (image-only) component
            loss = ce(logits)
        else:
            logits, logits_pub = self.net(x_full), self.net(x_pub)   # Feature-DP residual
            loss = ce(logits) - ce(logits_pub) if labels is not None else None

        out = {"logits": logits}
        if labels is not None:
            out["loss"] = loss
        return out


def accuracy_metric(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return {"accuracy": float((logits.argmax(-1) == labels).mean())}


# --------------------------------------------------------------------------- #
# 3. One arm = one call to get_trainer
# --------------------------------------------------------------------------- #
def run_arm(arm: str, train_ds, test_ds, args, out_root: str) -> float:
    use_dp = arm in {"dp-sgd", "feature-dp"}
    feature_dp = arm == "feature-dp"
    torch.manual_seed(args.seed)
    model = TwoModalityMLP(public_dim=train_ds.public.shape[1], use_private=(arm != "image-only"))

    # Under DP the logical batch is per_device_train_batch_size x gradient_accumulation_steps
    # (manual microbatching inside the DP trainer; see docs/mimic_cxr.md).
    microbatch, accum = args.microbatch, args.batch_size // args.microbatch
    training_args = TrainingArguments(
        output_dir=os.path.join(out_root, arm),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size if use_dp else 64,
        per_device_eval_batch_size=512,
        gradient_accumulation_steps=accum if use_dp else 1,
        learning_rate=args.lr_dp if use_dp else args.lr,
        optim="adamw_torch",
        logging_steps=50,
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        seed=args.seed,
        use_cpu=not torch.cuda.is_available(),
    )
    kwargs = dict(model=model, args=training_args, train_dataset=train_ds, data_collator=collate,
                  compute_metrics=accuracy_metric)
    if use_dp:
        logical_batch = args.batch_size
        kwargs["privacy_args"] = PrivacyArguments(
            # Clip norms are divided by the logical batch because the trainer clips
            # per-sample gradients of the microbatch-mean loss (same convention as train_mimic.py).
            per_sample_max_grad_norm=args.clip_norm / logical_batch,
            public_clip=args.public_clip_norm / logical_batch,
            target_epsilon=args.epsilon,
            target_delta=len(train_ds) ** -1.1,
            feature_dp=feature_dp,
        )
    trainer = get_trainer(use_dp=use_dp, loss_mode="decomposed" if use_dp else "standard", **kwargs)
    trainer.model_accepts_loss_kwargs = False
    trainer.train()
    acc = trainer.evaluate(test_ds, metric_key_prefix="test")["test_accuracy"]
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=["image-only", "both", "dp-sgd", "feature-dp"],
                    choices=["image-only", "both", "dp-sgd", "feature-dp"])
    ap.add_argument("--synthetic", action="store_true", help="Use Gaussian blobs instead of downloading MNIST.")
    ap.add_argument("--data-root", default=os.path.join(tempfile.gettempdir(), "medmap_toy_mnist"))
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--hint-noise", type=float, default=0.2, help="P(private hint is a wrong digit).")
    ap.add_argument("--public-resolution", type=int, default=7,
                    help="Side length of the average-pooled public image (28 = full resolution).")
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch-size", type=int, default=600, help="Logical batch (DP: expected batch).")
    ap.add_argument("--microbatch", type=int, default=100, help="Physical microbatch under DP.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-dp", type=float, default=5e-3)
    ap.add_argument("--epsilon", type=float, default=2.0)
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument("--public-clip-norm", type=float, default=50.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert args.batch_size % args.microbatch == 0, "--batch-size must be a multiple of --microbatch"

    if args.synthetic:
        (xtr, ytr), (xte, yte) = load_synthetic(args.n_train, args.n_test, args.seed)
    else:
        try:
            (xtr, ytr), (xte, yte) = load_mnist(args.data_root, args.n_train, args.n_test, args.seed)
        except Exception as exc:  # offline machines
            print(f"[toy] MNIST unavailable ({type(exc).__name__}); falling back to --synthetic data.")
            (xtr, ytr), (xte, yte) = load_synthetic(args.n_train, args.n_test, args.seed)
    train_ds = TwoModalityDataset(xtr, ytr, args.hint_noise, args.seed, args.public_resolution)
    test_ds = TwoModalityDataset(xte, yte, args.hint_noise, args.seed + 1, args.public_resolution)
    print(f"[toy] train={len(train_ds)} test={len(test_ds)} hint_noise={args.hint_noise} epsilon={args.epsilon}")

    results = {}
    with tempfile.TemporaryDirectory() as out_root:
        for arm in args.arms:
            results[arm] = run_arm(arm, train_ds, test_ds, args, out_root)
            print(f"[toy] {arm:<11s} test accuracy = {100 * results[arm]:.1f}%")

    print("\n[toy] Summary (test accuracy, %)")
    for arm, acc in results.items():
        tag = {"image-only": "floor: never sees x_priv", "both": "ceiling: non-private",
               "dp-sgd": f"protects everything, eps={args.epsilon}",
               "feature-dp": f"protects x_priv only, eps={args.epsilon}"}[arm]
        print(f"  {arm:<11s} {100 * acc:6.1f}   ({tag})")


if __name__ == "__main__":
    main()
