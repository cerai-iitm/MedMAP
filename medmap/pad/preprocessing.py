"""PAD-UFES-20 metadata preprocessing, download, and patient-level split.

The clinical metadata (age, gender, family background, smoking/drinking,
Fitzpatrick skin type, cancer history, lesion diameters, ...) is treated as
the *private* modality of the benchmark. Age and lesion diameters are
bucketed into clinical bins and every feature column is one-hot encoded
(missing values become an explicit "Unknown" category). Only the target
``diagnostic`` column is label encoded (6 classes: ACK, BCC, MEL, NEV, SCC,
SEK).

The train/val/test split is patient-level (StratifiedGroupKFold with
``groups=patient_id``): fold 0 is the test set, fold 1 the validation set,
and the remaining 8 folds form the training set (~80/10/10). It is fixed by
``split_seed`` independently of the run seed, so every run in a sweep sees
the identical split, and a hard patient-overlap check guards against leakage.
"""

import os
import random
import shutil
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------------------------------- #
# Constants (match the original notebook)
# --------------------------------------------------------------------------- #
NUM_CLASSES = 6
AGE_BINS = [0, 35, 45, 50, 55, 60, 65, 70, 75, 80, np.inf]
AGE_LABELS = ['0-35', '35-45', '45-50', '50-55', '55-60', '60-65',
              '65-70', '70-75', '75-80', '80+']

# Diameter bins: analogous to age bucketing; adjust here if a different
# clinical discretization is preferred.
DIAMETER_BINS = [0, 5, 10, 15, 20, 25, 30, np.inf]
DIAMETER_LABELS = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30+']

SUBGROUP_COLS = [
    'gender', 'background_father', 'background_mother', 'smoke', 'drink',
    'fitspatrick', 'skin_cancer_history', 'cancer_history'
]

# Official PAD-UFES-20 archive on Mendeley Data (CC BY 4.0).
MENDELEY_URL = "https://data.mendeley.com/public-api/zip/zr7vgbcyr2/download/1"


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Data download (optional — validates/repairs an existing partial extraction)
# --------------------------------------------------------------------------- #
def _missing_pad_images(metadata_path: str, img_dir: str) -> list[str]:
    """Return metadata image IDs that are absent from the flat image directory."""

    if not os.path.isfile(metadata_path):
        return ["<metadata.csv is missing>"]
    metadata = pd.read_csv(metadata_path, usecols=["img_id"])
    return [
        image_id
        for image_id in metadata["img_id"].astype(str)
        if not os.path.isfile(os.path.join(img_dir, image_id))
    ]


def download_and_extract(root: str, img_dir: str):
    """Download, extract, or repair PAD-UFES-20 into a flat layout."""

    os.makedirs(root, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    metadata_path = os.path.join(root, "metadata.csv")
    missing_before = _missing_pad_images(metadata_path, img_dir)
    if not missing_before:
        print(f"PAD-UFES-20 is complete at {root}; skipping download.")
        return
    if os.path.isfile(metadata_path) or os.listdir(img_dir):
        print(
            f"PAD-UFES-20 is partial ({len(missing_before)} missing entries); "
            "repairing it from the official archive."
        )

    local_filename = os.path.join(root, "PAD-UFES-20.zip")
    partial_filename = f"{local_filename}.part"
    if not os.path.isfile(local_filename):
        import requests

        print("Downloading PAD-UFES-20 dataset...")
        with requests.get(MENDELEY_URL, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with open(partial_filename, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        os.replace(partial_filename, local_filename)
    else:
        print(f"Using existing archive: {local_filename}")

    with ZipFile(local_filename, "r") as archive:
        archive.extractall(root)

    for part in ["imgs_part_1", "imgs_part_2", "imgs_part_3"]:
        part_archive = os.path.join(root, "images", f"{part}.zip")
        if not os.path.isfile(part_archive):
            raise FileNotFoundError(f"Expected image archive is missing: {part_archive}")
        with ZipFile(part_archive, "r") as archive:
            archive.extractall(root)

    for part in ["imgs_part_1", "imgs_part_2", "imgs_part_3"]:
        src = os.path.join(root, part)
        if not os.path.isdir(src):
            raise FileNotFoundError(f"Expected extracted image directory is missing: {src}")
        for filename in os.listdir(src):
            source = os.path.join(src, filename)
            destination = os.path.join(img_dir, filename)
            if os.path.isfile(source) and not os.path.exists(destination):
                shutil.move(source, destination)

    missing_after = _missing_pad_images(metadata_path, img_dir)
    if missing_after:
        examples = ", ".join(missing_after[:5])
        raise RuntimeError(
            f"PAD-UFES-20 extraction is incomplete: {len(missing_after)} metadata "
            f"images are missing. Examples: {examples}"
        )
    print(f"Done extracting and validating PAD-UFES-20 at {root}.")


# --------------------------------------------------------------------------- #
# Metadata preprocessing
# --------------------------------------------------------------------------- #
def _canonicalize_categorical(series: pd.Series) -> pd.Series:
    """Convert a metadata column to strings without label encoding."""
    def normalize(v):
        if pd.isna(v):
            return "Unknown"
        if isinstance(v, (bool, np.bool_)):
            return "True" if bool(v) else "False"
        text = str(v).strip()
        if text == "" or text.lower() in {"nan", "none"}:
            return "Unknown"
        if text.lower() == "true":
            return "True"
        if text.lower() == "false":
            return "False"
        if text.upper() == "BRASIL":
            return "BRAZIL"
        return text
    return series.map(normalize)


def bucket_age_label(age) -> str:
    if pd.isna(age) or age < 0:
        return "Unknown"
    bucket = np.searchsorted(AGE_BINS[1:-1], age, side='right')
    return AGE_LABELS[min(bucket, len(AGE_LABELS) - 1)]


def bucket_diameter_label(value) -> str:
    if pd.isna(value) or value < 0:
        return "Unknown"
    bucket = np.searchsorted(DIAMETER_BINS[1:-1], value, side='right')
    return DIAMETER_LABELS[min(bucket, len(DIAMETER_LABELS) - 1)]


def preprocess_metadata(path: str):
    """One-hot encode all metadata features; only the target is label encoded."""
    raw = pd.read_csv(path)

    required = {'patient_id', 'img_id', 'diagnostic'}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {sorted(missing)}")

    patient_ids = raw['patient_id'].astype(str).to_numpy()
    labels_raw = raw['diagnostic'].astype(str).to_numpy()
    img_ids = raw['img_id'].astype(str).to_numpy()

    # Raw values retained for subgroup reporting.
    subgroup_raw = {
        'age': raw['age'].copy(),
        'gender': _canonicalize_categorical(raw['gender']),
        'background_father': _canonicalize_categorical(raw['background_father']),
        'background_mother': _canonicalize_categorical(raw['background_mother']),
        'smoke': _canonicalize_categorical(raw['smoke']),
        'drink': _canonicalize_categorical(raw['drink']),
        'fitspatrick': _canonicalize_categorical(raw['fitspatrick']),
        'skin_cancer_history': _canonicalize_categorical(raw['skin_cancer_history']),
        'cancer_history': _canonicalize_categorical(raw['cancer_history']),
    }

    # patient_id, lesion_id and biopsed are excluded from model inputs.
    meta = raw.drop(columns=['patient_id', 'lesion_id', 'biopsed'], errors='ignore').copy()

    # Age/diameters are categorical bins, then everything is one-hot encoded.
    meta['age'] = meta['age'].map(bucket_age_label)
    meta['diameter_1'] = meta['diameter_1'].map(bucket_diameter_label)
    meta['diameter_2'] = meta['diameter_2'].map(bucket_diameter_label)

    feature_cols = [c for c in meta.columns if c not in ('img_id', 'diagnostic')]
    for col in feature_cols:
        meta[col] = _canonicalize_categorical(meta[col])

    ohe_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    meta_ohe = ohe_enc.fit_transform(meta[feature_cols]).astype(np.float32)

    # Target encoding is fine: diagnostic is the y variable, not a feature.
    le_label = preprocessing.LabelEncoder()
    y_all = le_label.fit_transform(labels_raw).astype(np.int64)

    print(f"One-hot metadata shape: {meta_ohe.shape}")
    print(f"One-hot metadata columns: {len(ohe_enc.get_feature_names_out(feature_cols))}")
    print("Classes:", le_label.classes_)

    return (img_ids, patient_ids, meta_ohe, y_all, le_label.classes_,
            subgroup_raw, ohe_enc.get_feature_names_out(feature_cols))


def age_to_quantile_ohe(age_series: pd.Series) -> np.ndarray:
    # Compatibility helper; current preprocessing uses OneHotEncoder on age bins.
    n = len(age_series)
    ohe_age = np.zeros((n, len(AGE_LABELS)), dtype=np.float32)
    for i, age in enumerate(age_series):
        label = bucket_age_label(age)
        if label in AGE_LABELS:
            ohe_age[i, AGE_LABELS.index(label)] = 1.0
    return ohe_age


# --------------------------------------------------------------------------- #
# Patient-level split
# --------------------------------------------------------------------------- #
def patient_stratified_split(patient_ids: np.ndarray, y_all: np.ndarray, split_seed: int):
    """Approximate 80/10/10 split with group-aware label stratification."""
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=split_seed)
    dummy_x = np.zeros((len(y_all), 1), dtype=np.float32)
    folds = [np.asarray(test_idx) for _, test_idx in
             sgkf.split(dummy_x, y_all, groups=patient_ids)]

    idx_te = folds[0]
    idx_va = folds[1]
    idx_tr = np.concatenate(folds[2:])
    return idx_tr, idx_va, idx_te


def print_split_class_distributions(split_indices, y_all, patient_ids, class_names):
    split_patient_sets = {}
    print("\n=== Patient-level split / class distributions ===")
    for split_name, idx in split_indices.items():
        patients = set(patient_ids[idx].tolist())
        split_patient_sets[split_name] = patients
        counts = np.bincount(y_all[idx], minlength=len(class_names))
        print(f"\n{split_name}: {len(idx)} samples, {len(patients)} unique patients")
        for cls, count in enumerate(counts):
            print(f"  {class_names[cls]:<5}: {count:4d} ({100*count/len(idx):5.2f}%)")

    names = list(split_patient_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = split_patient_sets[names[i]] & split_patient_sets[names[j]]
            if overlap:
                raise RuntimeError(
                    f"Patient leakage between {names[i]} and {names[j]}: {len(overlap)} patients."
                )
    print("\nPatient overlap check: PASS (0 patients shared across splits).")
