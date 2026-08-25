# Tiny ImageNet / ImageNet-Captions: images + captions (SmolVLM)

200-class image classification with SmolVLM
(`HuggingFaceTB/SmolVLM-256M-Base`, LoRA fine-tuning). Images are the public
modality; the Flickr-derived caption text (title + tags + description from
ImageNet-Captions) is the private modality. This non-medical arm sanity-checks
the privacy mechanisms on a standard vision-language task.

Contents: [Download](#1-download) · [Dataset assembly](#2-dataset-assembly) ·
[Training](#3-training) · [Parameter reference](#4-parameter-reference) · [License & citations](#5-license--citations)

---

## 1. Download

Three ingredients:

1. **ImageNet (ILSVRC 2012) training images** — from the Kaggle
   [ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge):

   ```bash
   pip install kaggle   # + place your API token at ~/.kaggle/kaggle.json
   kaggle competitions download -c imagenet-object-localization-challenge
   # keep the zip, or unzip it — the loader supports both:
   #   ROOT_ZIP_PATH=/data/imagenet_object_localization_challenge[.zip]
   ```

   Access requires accepting the ImageNet terms (non-commercial research /
   educational use).

2. **ImageNet-Captions JSON** — Flickr titles/tags/descriptions for ~450k
   ImageNet train images, from
   [mlfoundations/imagenet-captions](https://github.com/mlfoundations/imagenet-captions):

   ```bash
   git clone https://github.com/mlfoundations/imagenet-captions
   # -> imagenet_captions.json ; set CAPTIONS_JSON to its path
   ```

3. **Tiny ImageNet class list** — the 200-class subset is defined by
   `wnids.txt` from [tiny-imagenet-200](http://cs231n.stanford.edu/tiny-imagenet-200.zip):

   ```bash
   wget http://cs231n.stanford.edu/tiny-imagenet-200.zip && unzip tiny-imagenet-200.zip
   export TINY_IMAGENET_WNIDS=/data/tiny-imagenet-200/wnids.txt
   ```

## 2. Dataset assembly

`medmap/datasets/imagenet/builder.py` joins the captions JSON with the
ILSVRC train listing, builds `<image> <title> <tags> <description>` prompts,
and filters to the 200 Tiny ImageNet classes. Splitting is either stratified
60/20/20 by `SEED` (default) or, with `USE_SPLIT_IDX=1`, the fixed index
files packaged at `medmap/datasets/imagenet/splits/` (the exact split used
for the reported runs).

## 3. Training

This experiment family is preserved as-is from the original MedMAP
implementation (env-var configuration), with two fixes: `GRAD_CHECKPOINT=1`
no longer crashes (the gradient-checkpointing call now happens after model
creation) and the Tiny ImageNet class list is configurable via
`TINY_IMAGENET_WNIDS`.

```bash
export CAPTIONS_JSON=/data/imagenet_captions.json
export ROOT_ZIP_PATH=/data/imagenet_object_localization_challenge
export TINY_IMAGENET_WNIDS=/data/tiny-imagenet-200/wnids.txt

# Non-private (modalities via INCLUDE_IMG / INCLUDE_TEXT)
bash examples/run_imagenet_non_private.sh

# Full DP-SGD (FEATURE_DP=0) or Feature-DP (FEATURE_DP=1; image public,
# caption text private)
FEATURE_DP=1 TARGET_EPS=1.0 bash examples/run_imagenet_private.sh
```

Notable variables: `SEED` (100), `EPOCHS` (10), `BATCH`/`GRAD_ACCUM` (in DP
mode the per-device batch becomes `BATCH*GRAD_ACCUM`), `LR`, `MAX_LENGTH`
(128), `USE_LORA`/`LORA_R`/`LORA_ALPHA`/`LORA_DROPOUT`, `CLIP_NORM`,
`WANDB_MODE=disabled` to turn W&B off.

## 4. Parameter reference

### `scripts/train_imagenet.py` (environment variables)

**Data & model:**

| Variable | Default | Purpose |
| --- | --- | --- |
| `CAPTIONS_JSON` | — (edit example) | Path to `imagenet_captions.json`. |
| `ROOT_ZIP_PATH` | — (edit example) | ILSVRC download: either the Kaggle zip or its extracted directory. |
| `TRAIN_TXT` | `ILSVRC/ImageSets/CLS-LOC/train_cls.txt` | Train-split listing inside the ILSVRC layout. |
| `TINY_IMAGENET_WNIDS` | required | `wnids.txt` from tiny-imagenet-200 defining the 200-class subset. |
| `MODEL_NAME` | `HuggingFaceTB/SmolVLM-256M-Base` | Backbone checkpoint. |

**Modality & privacy:**

| Variable | Default | Purpose |
| --- | --- | --- |
| `INCLUDE_IMG` / `INCLUDE_TEXT` | 1 / 1 | Modality ablations (DP forces both on). |
| `USE_DP` | 0 | Switch differential privacy on. |
| `FEATURE_DP` | 0 | With `USE_DP=1`: 1 = Feature-DP (image public, caption text private), 0 = full DP-SGD. |
| `TARGET_EPS` / `CLIP_NORM` | 1.0 / 1.0 | Privacy budget and private clip norm (the public clip is `50/effective_batch`, matching the reported runs). |

**Optimization & splits:**

| Variable | Default | Purpose |
| --- | --- | --- |
| `SEED` | 100 | Run seed (also controls the random split when `USE_SPLIT_IDX=0`). |
| `USE_SPLIT_IDX` | 0 | 1 = use the fixed packaged split indices (`medmap/datasets/imagenet/splits/`), the exact split of the reported runs; 0 = stratified 60/20/20 split by `SEED`. |
| `EPOCHS` / `BATCH` / `GRAD_ACCUM` | 10 / 16 / 64 | Under DP the per-device batch becomes `BATCH*GRAD_ACCUM` (manual microbatching). |
| `LR` / `WEIGHT_DECAY` / `WARMUP_RATIO` | 5e-5 / 0.01 / 0.03 | AdamW settings (DP example uses lr 5e-4). |
| `MAX_LENGTH` | 128 | Caption truncation length. |
| `USE_LORA`, `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT` | 1, 16, 32, 0.05 | LoRA adapter configuration. |
| `GRAD_CHECKPOINT` | 0 | Gradient checkpointing (memory saver; safe to enable). |
| `EVAL_STRATEGY`, `EVAL_STEPS`, `SAVE_STRATEGY`, `SAVE_STEPS`, `LOGGING_STRATEGY`, `LOGGING_STEPS` | steps / 700 / steps / 700 / steps / 1 | Evaluation, checkpointing and logging cadence (keep save and eval aligned: best-checkpoint loading is on). |
| `OUT_DIR` / `RUN_NAME` | auto-generated from the setting | Output directory and W&B run name. |
| `WANDB_MODE` | online | `disabled` runs without W&B. |

---

## 5. License & citations

**Data terms:** ImageNet is available for non-commercial research and
educational purposes under its terms of access; Tiny ImageNet and
ImageNet-Captions derive from it. ImageNet-Captions metadata originates from
Flickr and is subject to the original uploaders' terms. No image or caption
data is redistributed with this repository.
**Model:** SmolVLM is released under Apache-2.0.

```bibtex
@inproceedings{deng2009imagenet,
  title     = {ImageNet: A large-scale hierarchical image database},
  author    = {Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
  booktitle = {CVPR}, pages = {248--255}, year = {2009}
}

@article{russakovsky2015ilsvrc,
  title   = {ImageNet Large Scale Visual Recognition Challenge},
  author  = {Russakovsky, Olga and Deng, Jia and Su, Hao and Krause, Jonathan and Satheesh, Sanjeev and Ma, Sean and Huang, Zhiheng and Karpathy, Andrej and Khosla, Aditya and Bernstein, Michael and others},
  journal = {International Journal of Computer Vision}, volume = {115}, pages = {211--252}, year = {2015}
}

@techreport{le2015tinyimagenet,
  title       = {Tiny ImageNet Visual Recognition Challenge},
  author      = {Le, Ya and Yang, Xuan},
  institution = {Stanford CS231N}, year = {2015}
}

@inproceedings{fang2022imagenetcaptions,
  title     = {Data determines distributional robustness in contrastive language image pre-training (CLIP)},
  author    = {Fang, Alex and Ilharco, Gabriel and Wortsman, Mitchell and Wan, Yuhao and Shankar, Vaishaal and Dave, Achal and Schmidt, Ludwig},
  booktitle = {International Conference on Machine Learning (ICML)}, year = {2022},
  note      = {source of ImageNet-Captions}
}

@article{marafioti2025smolvlm,
  title   = {SmolVLM: Redefining small and efficient multimodal models},
  author  = {Marafioti, Andr{\'e}s and Zohar, Orr and Farr{\'e}, Miquel and Noyan, Merve and Bakouch, Elie and Cuenca, Pedro and Zakka, Cyril and Ben Allal, Loubna and Lozhkov, Anton and Tazi, Nouamane and others},
  journal = {arXiv preprint arXiv:2504.05299}, year = {2025}
}
```
