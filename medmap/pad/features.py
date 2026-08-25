"""PAD-UFES-20 image features (frozen EfficientNet-B3) and feature assembly.

Images are the *public* modality: a frozen ImageNet-pretrained
EfficientNet-B3 backbone (features + avgpool, eval mode) turns each lesion
photo into a 1536-d embedding. Extraction runs once and is cached to an
``.npz`` file so seed/epsilon sweeps skip it.

``combine_features`` concatenates ``[image features | metadata one-hot]``.
With ``use_metadata=False`` the metadata block is replaced by a zero vector
of the same width, so input dimensionality is constant across settings.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

from medmap.pad.preprocessing import download_and_extract, preprocess_metadata

IMG_SIZE = 300  # EfficientNet-B3 native input resolution

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


def build_feature_extractor(device) -> nn.Module:
    backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    feature_extractor = nn.Sequential(backbone.features, backbone.avgpool)
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor.eval()
    return feature_extractor.to(device)


def extract_image_features(meta: pd.DataFrame, feature_extractor: nn.Module,
                            img_dir: str, device, batch_size: int = 32) -> np.ndarray:
    img_ids = list(meta['img_id'])
    all_feats = []
    feature_extractor.eval()
    with torch.no_grad():
        for start in range(0, len(img_ids), batch_size):
            batch_ids = img_ids[start:start + batch_size]
            tensors = [IMG_TRANSFORM(Image.open(os.path.join(img_dir, i)).convert("RGB"))
                       for i in batch_ids]
            batch = torch.stack(tensors).to(device)
            feat = feature_extractor(batch)
            feat = feat.view(feat.size(0), -1)
            all_feats.append(feat.cpu().numpy())
            if (start // batch_size) % 10 == 0:
                print(f"  Extracted {min(start + batch_size, len(img_ids))}/{len(img_ids)}")
    return np.concatenate(all_feats, axis=0).astype(np.float32)


def load_or_build_raw_features(cache_path: str, meta_path: str, img_dir: str,
                                root: str, download: bool, device):
    """Rebuild metadata preprocessing every run; cache only image features."""
    if download and not (os.path.isdir(img_dir) and len(os.listdir(img_dir)) > 0):
        download_and_extract(root, img_dir)

    img_ids, patient_ids, meta_ohe, y_all, class_names, subgroups, meta_feature_names = (
        preprocess_metadata(meta_path)
    )

    img_features = None
    if os.path.exists(cache_path):
        print(f"Loading cached image features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        if 'img_features' in data.files and len(data['img_features']) == len(img_ids):
            img_features = data['img_features'].astype(np.float32)
            print("Using cached EfficientNet features.")
        else:
            print("Cache incompatible with current metadata; rebuilding image features.")

    if img_features is None:
        feature_extractor = build_feature_extractor(device)
        img_meta = pd.DataFrame({'img_id': img_ids})
        img_features = extract_image_features(
            img_meta, feature_extractor, img_dir, device, batch_size=32
        ).astype(np.float32)
        np.savez(cache_path, img_features=img_features)
        print(f"Cached image features to {cache_path}")

    if len(img_features) != len(y_all):
        raise ValueError(
            f"Image features ({len(img_features)}) and metadata ({len(y_all)}) differ in row count."
        )

    return (img_features, meta_ohe, y_all, img_features.shape[1], class_names,
            subgroups, patient_ids, meta_feature_names)


def combine_features(img_features: np.ndarray, meta_ohe: np.ndarray, use_metadata: bool) -> np.ndarray:
    """
    use_metadata=True  -> [image features | real metadata one-hot]
    use_metadata=False -> [image features | zero vector of the usual metadata size]
                           ("no meta" setting — same input dimensionality, metadata
                           block just carries no information)
    """
    meta_block = meta_ohe if use_metadata else np.zeros_like(meta_ohe)
    return np.concatenate([img_features, meta_block], axis=1).astype(np.float32)


class CombinedDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
