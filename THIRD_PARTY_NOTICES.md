# Third-party notices

MedMAP's own code is released under the MIT License (see `LICENSE`).
This repository additionally redistributes the third-party components
listed below. Each component remains under its original license, which is
retained alongside the code.

## 1. dp-transformers (modified) — `library/dp-transformers/`

- Upstream: <https://github.com/microsoft/dp-transformers>
- License: MIT (Copyright (c) Microsoft Corporation) — retained at
  `library/dp-transformers/LICENSE.txt`
- Status: **modified copy** (base: upstream commit `f9fae44`, v1.0.1).
  The modifications implement the Feature-DP mechanism used
  throughout MedMAP. A complete change log is provided in
  `library/dp-transformers/MODIFICATIONS.md`.
- Citation: if you use the DP training machinery, please also acknowledge
  dp-transformers as requested by upstream:

  ```bibtex
  @misc{dp-transformers,
    author        = {Wutschitz, Lukas and Inan, Huseyin A. and Manoel, Andre},
    title         = {dp-transformers: Training transformer models with differential privacy},
    year          = {2022},
    month         = {August},
    howpublished  = {\url{https://www.microsoft.com/en-us/research/project/dp-transformers}}
  }
  ```

## 2. Hugging Face Transformers `Trainer` (vendored, modified) —
`library/dp-transformers/src/dp_transformers/custom_trainer.py`

- Upstream: <https://github.com/huggingface/transformers>
- License: Apache License 2.0 (Copyright 2020-present, the Hugging Face
  Team)
- Status: **modified copy** of `transformers.Trainer`. Per Apache-2.0
  §4(b), we state that this file has been changed: it adds a
  `public_batch` constructor argument and threads `public=` /
  `feature_dp=` keyword arguments through the training loop and
  `compute_loss` (see `library/dp-transformers/MODIFICATIONS.md`).
- The full Apache License 2.0 text is available at
  <https://www.apache.org/licenses/LICENSE-2.0>.

## 3. MIT-LCP mimic-code section parser (vendored, unmodified) —
`medmap/datasets/mimic_cxr/vendor/mimic_code/`

- Upstream: <https://github.com/MIT-LCP/mimic-code>
  (pinned revision recorded in `medmap/datasets/mimic_cxr/vendor/SOURCES.md`)
- License: MIT — retained beside the vendored file.
- Used to reproduce the official MIMIC-CXR section segmentation when
  removing the Findings / Impression sections from free-text reports.
- Citation:

  ```bibtex
  @article{johnson2018mimic_code,
    title   = {The MIMIC Code Repository: enabling reproducibility in critical care research},
    author  = {Johnson, Alistair E W and Stone, David J and Celi, Leo A and Pollard, Tom J},
    journal = {Journal of the American Medical Informatics Association},
    volume  = {25}, number = {1}, pages = {32--39}, year = {2018}
  }
  ```

## 4. CheXpert labeler phrase resources (vendored, unmodified) —
`medmap/datasets/mimic_cxr/vendor/chexpert_labeler/`

- Upstream: <https://github.com/stanfordmlgroup/chexpert-labeler>
  (pinned revision recorded in `vendor/SOURCES.md`)
- License: MIT — retained beside the vendored resources.
- The official mention/unmention phrase lists are used to mask
  label-bearing phrases in report text (leakage filtering).
- Citation:

  ```bibtex
  @inproceedings{irvin2019chexpert,
    title     = {CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison},
    author    = {Irvin, Jeremy and Rajpurkar, Pranav and Ko, Michael and Yu, Yifan and Ciurea-Ilcus, Silviana and Chute, Chris and Marklund, Henrik and Haghgoo, Behzad and Ball, Robyn and Shpanskaya, Katie and others},
    booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
    volume    = {33}, pages = {590--597}, year = {2019}
  }
  ```

## 5. NegBio CheXpert patterns (vendored, unmodified) —
`medmap/datasets/mimic_cxr/vendor/negbio/`

- Upstream: <https://github.com/ncbi-nlp/NegBio> (MIMIC-CXR branch;
  pinned revision recorded in `vendor/SOURCES.md`)
- License: retained beside the vendored resources (public-domain /
  NCBI license terms as distributed by upstream).
- The include/exclude regular-expression patterns are used as the second
  matcher in leakage filtering.
- Citation:

  ```bibtex
  @inproceedings{peng2018negbio,
    title     = {NegBio: a high-performance tool for negation and uncertainty detection in radiology reports},
    author    = {Peng, Yifan and Wang, Xiaosong and Lu, Le and Bagheri, Mohammadhadi and Summers, Ronald and Lu, Zhiyong},
    booktitle = {AMIA Summits on Translational Science Proceedings},
    pages     = {188--196}, year = {2018}
  }
  ```

## 6. Datasets (NOT redistributed)

No dataset content (images, reports, phenotypes, metadata, or labels) is
included in this repository. Access is governed by each dataset's own
license / data use agreement; see the per-dataset pages under `docs/` for
download instructions, license terms, and required citations:

| Dataset | License / access | Doc |
| --- | --- | --- |
| MIMIC-CXR & MIMIC-CXR-JPG | PhysioNet Credentialed Health Data License 1.5.0 (credentialed access + CITI training) | `docs/mimic_cxr.md` |
| ABIDE I Preprocessed (PCP) | Creative Commons BY-NC-SA 3.0 | `docs/abide.md` |
| PAD-UFES-20 | Creative Commons BY 4.0 | `docs/pad_ufes.md` |
| ImageNet / Tiny ImageNet / ImageNet-Captions | ImageNet terms of access (non-commercial research) | `docs/imagenet.md` |

## 7. Pretrained models (downloaded at run time, NOT redistributed)

| Model | Source | Terms |
| --- | --- | --- |
| Gemma 3 4B (`google/gemma-3-4b-pt`) | Hugging Face Hub | Gemma Terms of Use |
| MedGemma 4B (`google/medgemma-4b-pt`) | Hugging Face Hub | Health AI Developer Foundations terms of use |
| SmolVLM | Hugging Face Hub | Apache License 2.0 |
| EfficientNet-B3 (ImageNet weights) | torchvision | BSD 3-Clause (torchvision) |
