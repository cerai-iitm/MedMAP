# Modifications relative to upstream microsoft/dp-transformers

This directory contains a **modified copy** of
[microsoft/dp-transformers](https://github.com/microsoft/dp-transformers)
(MIT License, Copyright (c) Microsoft Corporation), based on upstream
commit `f9fae44` ("Add automatic testing of examples (#36)", v1.0.1).

It is the differential-privacy backbone of MedMAP: all DP-SGD and
Feature-DP fine-tuning of the deep models
(Gemma 3 / MedGemma, SmolVLM, and the ABIDE Hugging Face linear model) runs through
the `OpacusDPTrainer` in this fork. The modifications implement the
Feature-DP mechanism: per-sample gradients are decomposed into a *public*
component (computed with the sensitive modality removed) and a *private
residual* component, which are clipped separately; only the private
component determines the noise scale.

> **Naming note:** earlier versions of this fork called the mechanism
> `selective_dp`. In MedMAP the flag/field is uniformly named `feature_dp`
> (`PrivacyArguments.feature_dp`, the optimizer flag, and the `feature_dp=`
> keyword threaded through the trainer and model forwards). Keep this in
> mind when diffing against the pre-MedMAP fork history.

## Summary of changes

### Added files (`src/dp_transformers/`)

| File | Purpose |
| --- | --- |
| `custom_trainer.py` | Vendored copy of the Hugging Face `transformers.Trainer` (Apache License 2.0, see THIRD_PARTY_NOTICES.md at the repository root) extended with a `public_batch` constructor argument. When enabled, the training loop draws, for every private batch, an independently shuffled *public* batch of the same size from a second dataloader (two batches per step, as in Algorithm 3 of the paper), and `compute_loss` forwards `public=` / `feature_dp=` keyword arguments into the model call. |
| `custom_dp_optimizer.py` | Reimplementation of Opacus's `DPOptimizer` with a `feature_dp` flag and `gradient_accumulation_steps` support. Per-sample clipping is performed in the trainer (the optimizer's clip factor is identity); `add_noise` adds Gaussian noise with `std = noise_multiplier * max_grad_norm`. |
| `custom_ddp_optimizer.py` | Distributed (DDP) variant of the above. |
| `custom_adam.py` | `CustomAdamW`: an AdamW variant that receives buffered *public* gradients separately from the noised *private* gradients in Feature-DP mode, so that only the private component is privatized. |

### Modified files

| File | Change |
| --- | --- |
| `arguments.py` | `PrivacyArguments` gains two fields: `public_clip` (max per-sample clip norm for the public gradient component) and `feature_dp` (enables Feature-DP). `find_noise_multiplier` was rewritten to calibrate the noise multiplier with Google's [`dp-accounting`](https://github.com/google/differential-privacy/tree/main/python/dp_accounting) RDP accountant (`PoissonSampledDpEvent` composed over the number of steps) solved with `scipy.optimize.brentq`, replacing the upstream `prv_accountant`-based implementation. |
| `dp_utils.py` | `OpacusDPTrainer` now inherits from the local `custom_trainer.Trainer` and implements manual microbatching (`per_device_train_batch_size` must be divisible by `gradient_accumulation_steps`; HF gradient accumulation is disabled). The rewritten `training_step` computes per-sample gradients on the private batch, optionally recomputes a public-only forward/backward (`public=True`), clips public and private per-sample gradients separately (`public_clip` / `per_sample_max_grad_norm`), and either merges them or (with `CustomAdamW`) buffers the public component so noise is applied only to the private component. The dead `prv_accountant` import was removed. |
| `setup.py` | Version marked `1.0.1+medmap`; `prv_accountant` dependency replaced with `dp-accounting` + `scipy`. |

### Removed relative to upstream

Upstream `examples/`, `research/`, `tests/`, and CI workflow files are not
included in this vendored copy.

## License

The original MIT license (`LICENSE.txt`, Copyright (c) Microsoft
Corporation) is retained and applies to this modified copy.
`custom_trainer.py` contains code derived from Hugging Face Transformers,
which is licensed under the Apache License 2.0; as required by that
license, this file documents that the code was changed (the `public_batch`
data flow described above).

## Installation

This copy is used directly from the source tree. Either rely on the
repository's `scripts/_bootstrap.py` (adds `library/dp-transformers/src`
to `sys.path`), or install it as a package:

```bash
pip install -e library/dp-transformers
```
