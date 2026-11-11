# Tiny ImageNet / ImageNet-Captions: images + captions (SmolVLM)

This task classifies the 200 Tiny ImageNet classes with full-resolution ILSVRC
training images and SmolVLM (`HuggingFaceTB/SmolVLM-256M-Base`). Images are
the public modality; Flickr-derived ImageNet-Captions text (title, tags, and
description) is the private modality.

See [data_layout.md](data_layout.md) for the shared workspace and root
overrides.

## 1. Access and download

Three sources are required:

1. **ILSVRC 2012 training images** from the Kaggle
   [ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge).
   Log in, accept the competition/ImageNet terms, install the Kaggle CLI, and
   put its API credential at `~/.kaggle/kaggle.json` with mode `600`.
2. **ImageNet-Captions** from the official
   [mlfoundations repository](https://github.com/mlfoundations/imagenet-captions).
3. **Tiny ImageNet class IDs** (`wnids.txt`) from
   [tiny-imagenet-200](http://cs231n.stanford.edu/tiny-imagenet-200.zip).
   The benchmark uses its class list, not the 64×64 images.

After accepting the required terms:

```bash
bash scripts/setup_workspace.sh
bash scripts/download_tiny_imagenet.sh
```

The helper downloads all three sources and creates:

```text
data/raw/tiny-imagenet/
├── ilsvrc/
│   └── imagenet-object-localization-challenge.zip
│       # or extracted ILSVRC/ImageSets/... and ILSVRC/Data/...
├── captions/imagenet-captions/imagenet_captions.json
└── tiny-imagenet-200/wnids.txt
```

If ILSVRC came from another approved source, place the ZIP or extracted
`ILSVRC/` tree under `data/raw/tiny-imagenet/ilsvrc/`, then download only the
two public metadata sources:

```bash
bash scripts/download_tiny_imagenet.sh --skip-ilsvrc
```

No ImageNet image or caption data is redistributed by MedMAP.

## 2. Validate and assemble

The builder joins `imagenet_captions.json` to
`ILSVRC/ImageSets/CLS-LOC/train_cls.txt`, creates
`<image> <title> <tags> <description>` prompts, and filters to the 200 WNIDs.
Before model download/training, validate the entire join and packaged split:

```bash
bash examples/run_imagenet_non_private.sh --validate-only
```

Validation checks:

- the captions JSON, ILSVRC directory/ZIP, train listing, and WNID file exist;
- `wnids.txt` contains exactly 200 unique IDs;
- the source join contains all 200 classes; and
- packaged train/validation/test indices are integer, in bounds, unique, and
  mutually disjoint.

The reported experiments use packaged fixed indices (`--use-fixed-split`).
Use `--no-use-fixed-split` only for a new seed-stratified 60/20/20 split.

## 3. Training

Launchers provide all data/cache/output paths and forward every extra option
to `scripts.train_imagenet`:

```bash
# Non-private image + caption
bash examples/run_imagenet_non_private.sh \
  --epochs 10 --batch-size 32 --learning-rate 5e-5

# Non-private modality ablations
bash examples/run_imagenet_non_private.sh --include-img --no-include-text
bash examples/run_imagenet_non_private.sh --no-include-img --include-text

# Feature-DP (default private mechanism) and full DP-SGD
IMAGENET_PRIVACY_MODE=feature_dp TARGET_EPSILON=4 \
  bash examples/run_imagenet_private.sh --gradient-checkpointing
IMAGENET_PRIVACY_MODE=full_dp TARGET_EPSILON=4 \
  bash examples/run_imagenet_private.sh
```

DP forces both modalities. As in the reported implementation, DP expands the
per-device batch to `batch-size × gradient-accumulation-steps` for manual
microbatching in the modified trainer; reduce these values if the requested
effective batch does not fit the available GPU.

W&B is disabled by default. Enable it deliberately with
`--wandb-mode online`; optionally pass `--wandb-project` and
`--wandb-entity`. Outputs are placed under `outputs/tiny-imagenet/` and now
include an explicit final test metrics file after best-checkpoint restoration.

## 4. CLI reference

Run `python -m scripts.train_imagenet --help` for the authoritative list.
Environment variables from the legacy implementation remain fallback values,
but explicit CLI options take precedence.

| Option | Default | Purpose |
| --- | --- | --- |
| `--captions-json` | required | `imagenet_captions.json`. |
| `--root-zip-path` | required | Kaggle competition ZIP or extracted root containing `ILSVRC/`. |
| `--train-txt` | `ILSVRC/ImageSets/CLS-LOC/train_cls.txt` | Train listing inside the directory/ZIP. |
| `--tiny-imagenet-wnids` | required | Tiny ImageNet `wnids.txt`. |
| `--model-name` | `HuggingFaceTB/SmolVLM-256M-Base` | Backbone ID or local path. |
| `--output-dir`, `--run-name` | generated | Artifact directory and run label. |
| `--validate-only` | off | Assemble/validate data and split, then exit before model loading. |
| `--include-img`, `--include-text` | on, on | Non-private modalities; at least one is required. |
| `--use-dp` | off | Enable differential privacy; forces both modalities. |
| `--feature-dp` | off | With DP, Feature-DP when enabled and full DP-SGD when disabled. |
| `--target-epsilon`, `--clip-norm`, `--public-clip-norm` | 1, 1, 50 | Privacy budget and private/public clipping. |
| `--use-fixed-split` | on | Use packaged reported split indices. |
| `--seed` | 100 | Model/data seed (also random-split seed when fixed split is off). |
| `--epochs`, `--batch-size`, `--gradient-accumulation-steps` | 10, 16, 64 | Optimization schedule/microbatching. |
| `--learning-rate`, `--weight-decay`, `--warmup-ratio` | 5e-5, 0.01, 0.03 | AdamW settings. |
| `--max-length` | 128 | Caption token limit. |
| `--use-lora`, `--lora-r`, `--lora-alpha`, `--lora-dropout` | on, 16, 32, 0.05 | LoRA configuration. |
| `--gradient-checkpointing` | off | Trade compute for memory. |
| `--eval-strategy`, `--eval-steps` | steps, 700 | Validation cadence. |
| `--save-strategy`, `--save-steps` | steps, 700 | Checkpoint cadence; strategy must match evaluation when best-model loading is on. |
| `--resume-from-checkpoint` | unset | Resume an interrupted Trainer run. |
| `--wandb-mode` | disabled | `disabled`, `offline`, or `online`. |

## 5. Terms and citations

**Data terms:** ImageNet is available for non-commercial research and
educational purposes under its terms of access; Tiny ImageNet and
ImageNet-Captions derive from it. ImageNet-Captions metadata originates from
Flickr and remains subject to original uploader/source terms.
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
