"""MedMAP: a benchmark for multimodal medical machine learning under
differential privacy.

MedMAP evaluates non-private training, DP-SGD, and Feature-DP (Feature-DP
that clips/noises only the gradient contribution of the sensitive modality)
across four datasets:

- MIMIC-CXR (chest X-rays + leakage-filtered reports, MedGemma backbone)
- ABIDE I (fMRI functional connectivity + phenotypes, logistic regression;
  both a classic sklearn/NumPy backend and a Hugging Face backend)
- PAD-UFES-20 (skin-lesion images + clinical metadata, EfficientNet features)
- Tiny ImageNet / ImageNet-Captions (images + captions, SmolVLM backbone)

Subpackages
-----------
medmap.common        Shared utilities (seeding, env parsing, DP accounting).
medmap.datasets      Dataset download, preprocessing, and builder modules.
medmap.models        Model wrappers with a unified forward contract
                     (``public=``, ``feature_dp=``, ``loss_mode=``) so any
                     model can be dropped into the same trainers.
medmap.training      Trainers (standard + DP), collators, callbacks, metrics.
medmap.abide_classic ABIDE classic backend: sklearn LBFGS baseline plus
                     NumPy/Numba full-batch DP-GD and Feature-DP.
medmap.pad           PAD-UFES-20 pipeline (features, DP training loops).
"""

__version__ = "1.0.0"
