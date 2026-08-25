"""PAD-UFES-20 pipeline: skin-lesion images (public modality, frozen
EfficientNet-B3 features) + clinical metadata (private modality, one-hot).

Modules:
    preprocessing  download, metadata cleaning/one-hot, patient-level split
    features       EfficientNet-B3 feature extraction + on-disk cache
    training       non-private loop + shared evaluation
    privacy        DP-SGD and Feature DP-SGD loops + RDP noise calibration
    evaluation     test metrics + per-subgroup accuracy CSV

Entry point: scripts/train_pad.py (--mode sgd | dpsgd | feature_dpsgd).
The model lives in medmap.models.pad_mlp and is swappable: any nn.Module
mapping the combined feature vector to class logits works with these loops.
"""
