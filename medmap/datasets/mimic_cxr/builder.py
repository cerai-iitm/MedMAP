"""MIMIC-CXR + CheXpert dataset builder with on-disk caching.

Builds a Hugging Face ``DatasetDict`` (train/validation/test, official
PhysioNet split) from:

- the per-shard best-image index written by
  :mod:`medmap.datasets.mimic_cxr.image_selection`
  (``<index_root>/<shard>/<shard>_best_img_final.csv``),
- the leakage-filtered reports written by
  :mod:`medmap.datasets.mimic_cxr.report_extraction`
  (``s<study_id>_mmp.txt`` siblings + per-shard audit CSV), and
- the official CheXpert label CSV and split CSV distributed with
  MIMIC-CXR-JPG.

Studies listed in the audit CSV (no usable Findings/Impression section) are
excluded. CheXpert convention: ``-1`` (uncertain) is converted to NaN so it
is ignored by the masked BCE loss; the ``Support Devices`` and ``No Finding``
columns are dropped, leaving 12 label columns.

The result is cached to disk keyed by a fingerprint of every source file
(path, size, mtime), so repeated runs (seed/epsilon sweeps) skip the build.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk
from datasets import Image as DatasetsImage

from medmap.datasets.mimic_cxr.report_extraction import AUDIT_CSV_NAME, REPORT_SUFFIX

# Filename suffix of the leakage-filtered reports consumed by the builder.
REPORT_FILENAME_SUFFIX = f"{REPORT_SUFFIX}.txt"


def _image_to_path(image_field) -> str:
    # HF datasets.Image can be str path or dict with "path"
    if isinstance(image_field, str):
        return image_field
    if isinstance(image_field, dict) and isinstance(image_field.get("path"), str):
        return image_field["path"]
    raise TypeError(f"Unsupported image field type: {type(image_field)}")


def _parse_ids_from_image_path(image_path: str):
    # .../files/p10/p10000032/s50414267/<dicom_id>.jpg
    parts = [p for p in str(image_path).replace("\\", "/").split("/") if p]
    if len(parts) < 3:
        raise ValueError(f"Bad image_path (too short): {image_path}")

    dicom_file = parts[-1]
    study_dir = parts[-2]
    subj_dir = parts[-3]

    if not subj_dir.startswith("p") or len(subj_dir) <= 3:
        raise ValueError(f"Bad subject dir '{subj_dir}' from: {image_path}")
    if not study_dir.startswith("s") or len(study_dir) <= 2:
        raise ValueError(f"Bad study dir '{study_dir}' from: {image_path}")

    subject_id = int(subj_dir[1:])
    study_id = int(study_dir[1:])

    dicom_id, _ = os.path.splitext(dicom_file)
    if not dicom_id:
        raise ValueError(f"Bad dicom filename '{dicom_file}' from: {image_path}")

    return dicom_id, subject_id, study_id


def split_mimic_dataset_official_by_ids(
    ds: Dataset,
    split_csv_path: str,
    strict: bool = True,
    keep_ids: bool = True,
) -> DatasetDict:
    """
    Split HF Dataset into train/validation/test using official PhysioNet split CSV,
    matching on triple key (dicom_id, subject_id, study_id).

    Assumes ds already has columns: dicom_id (str), subject_id (int), study_id (int).
    """

    # 1) Load official split CSV
    split_df = pd.read_csv(split_csv_path, usecols=["dicom_id", "subject_id", "study_id", "split"])
    split_df["dicom_id"] = split_df["dicom_id"].astype(str)
    split_df["subject_id"] = split_df["subject_id"].astype("int64")
    split_df["study_id"] = split_df["study_id"].astype("int64")
    split_df["split"] = split_df["split"].astype(str)

    # 2) Guard against conflicting duplicate keys
    key_cols = ["dicom_id", "subject_id", "study_id"]
    dup = split_df.duplicated(subset=key_cols, keep=False)
    if dup.any():
        nunique = split_df[dup].groupby(key_cols)["split"].nunique()
        bad = nunique[nunique > 1]
        if len(bad) > 0:
            ex = split_df.merge(bad.reset_index(name="nuniq"), on=key_cols, how="inner").head(20)
            raise ValueError(f"Conflicting split assignments for same triple key. Example:\n{ex}")
        split_df = split_df.drop_duplicates(subset=key_cols, keep="first")

    # 3) Build lookup dict for O(1) split assignment
    split_map = {
        (r["dicom_id"], int(r["subject_id"]), int(r["study_id"])): r["split"]
        for _, r in split_df.iterrows()
    }

    # 4) Add __split by looking up triple key from ds
    def add_split(ex):
        key = (str(ex["dicom_id"]), int(ex["subject_id"]), int(ex["study_id"]))
        return {"__split": split_map.get(key, None)}

    ds2 = ds.map(add_split, desc="Assign official split via (dicom_id,subject_id,study_id)")

    # 5) Handle missing matches
    missing = ds2.filter(lambda x: x["__split"] is None)
    n_missing = len(missing)
    if n_missing > 0:
        msg = f"[split] {n_missing} rows could not be matched to split CSV using (dicom_id,subject_id,study_id)."
        if strict:
            k = min(5, n_missing)
            raise ValueError(msg + f" First {k} unmatched:\n{missing.select(range(k))}")
        else:
            print(msg + " Dropping them (strict=False).")
            ds2 = ds2.filter(lambda x: x["__split"] is not None)

    # 6) Create splits
    train_ds = ds2.filter(lambda x: x["__split"] == "train")
    val_ds = ds2.filter(lambda x: x["__split"] == "validate")
    test_ds = ds2.filter(lambda x: x["__split"] == "test")

    # 7) Cleanup helper and optionally drop id columns
    drop_cols = ["__split"]
    if not keep_ids:
        drop_cols += ["dicom_id", "subject_id", "study_id"]

    train_ds = train_ds.remove_columns(drop_cols)
    val_ds = val_ds.remove_columns(drop_cols)
    test_ds = test_ds.remove_columns(drop_cols)

    return DatasetDict(train=train_ds, validation=val_ds, test=test_ds)


def _resolve_indexed_image_path(image_path: str, image_root: str | None) -> str:
    """Resolve a portable index path against the explicit JPG version root."""

    path = Path(str(image_path))
    if path.is_absolute():
        resolved = path
    else:
        if image_root is None:
            raise ValueError(
                "The best-image index contains relative paths, but no image_root was "
                "provided. Pass the MIMIC-CXR-JPG version directory through "
                "--image_root."
            )
        resolved = Path(image_root).resolve() / path
    if not resolved.is_file():
        raise FileNotFoundError(f"Indexed MIMIC-CXR JPG does not exist: {resolved}")
    return str(resolved)


def _read_required_report(path: str) -> str:
    """Read one processed report and fail clearly instead of returning empty text."""

    report_path = Path(path)
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Processed MIMIC-CXR report does not exist: {report_path}. "
            "Run scripts/prepare_mimic_cxr.sh before training."
        )
    return report_path.read_text(encoding="utf-8")


def build_mimic_chexpert_dataset(
    top_level_buckets: List[str],
    best_img_index_root: str,
    chexpert_csv: str,
    report_root: str,
    image_root: str | None = None,
    drop_all_nan_rows: bool = True,
) -> Tuple[Dataset, int, List[str]]:
    """
    Builds an HF Dataset with columns:
      - text: leakage-filtered report text (may be empty if report missing)
      - image: datasets.Image (path stored; decoded on-the-fly)
      - labels: list[float32] with {0.0, 1.0, NaN} (NaNs ignored in loss)

    Notes:
    - CheXpert convention: -1 = uncertain -> convert to NaN so it is ignored in loss.
    """
    best_df_concat = None
    for top_level_bucket in top_level_buckets:
        best_csv_path = os.path.join(
            best_img_index_root, top_level_bucket, f"{top_level_bucket}_best_img_final.csv"
        )
        # 1) Base CSV with best image per study
        best_df = pd.read_csv(best_csv_path, usecols=['image_path', 'subject_id', 'study_id', 'dicom_id'])

        best_df['subject_id'] = best_df['subject_id'].astype(int)
        best_df['study_id'] = best_df['study_id'].astype(int)
        best_df['dicom_id'] = best_df['dicom_id'].astype(str)
        best_df['image_path'] = best_df['image_path'].apply(
            lambda value: _resolve_indexed_image_path(value, image_root)
        )
        print(f"Loaded {len(best_df)} entries from {best_csv_path}")

        excluded_studies_csv = os.path.join(report_root, top_level_bucket, AUDIT_CSV_NAME)
        excluded_study_ids = pd.read_csv(excluded_studies_csv, usecols=["study_id"])["study_id"].astype(int)
        best_df = best_df[~best_df["study_id"].isin(excluded_study_ids)].copy()
        print(f"Kept {len(best_df)} entries after excluding study IDs from {excluded_studies_csv}")

        # Build report paths:
        # <report_root>/<bucket>/p<subject_id>/s<study_id>_mmp.txt
        best_df["report_path"] = (
            f"{report_root}/{top_level_bucket}"
            + "/p"
            + best_df['subject_id'].astype(str)
            + "/s"
            + best_df['study_id'].astype(str)
            + REPORT_FILENAME_SUFFIX
        )
        best_df["report_txt"] = best_df["report_path"].apply(_read_required_report)

        # Concatenate
        if best_df_concat is None:
            best_df_concat = best_df
        else:
            best_df_concat = pd.concat([best_df_concat, best_df], ignore_index=True)
    print(f"Total combined entries from all buckets: {len(best_df_concat)}")

    chexpert_df = pd.read_csv(chexpert_csv)
    chexpert_df["subject_id"] = chexpert_df["subject_id"].astype(int)
    chexpert_df["study_id"] = chexpert_df["study_id"].astype(int)

    # Only keep studies we have in best_df
    chexpert_df = chexpert_df[chexpert_df["study_id"].isin(best_df_concat["study_id"].unique())].copy()

    id_cols = ["subject_id", "study_id"]
    label_cols = [c for c in chexpert_df.columns if c not in id_cols and c not in ["Support Devices", "No Finding"]]

    merged_df = best_df_concat.merge(
        chexpert_df[id_cols + label_cols],
        on=id_cols,
        how="left",
        validate="one_to_one",
    )

    merged_df[label_cols] = merged_df[label_cols].replace(-1, np.nan)

    if drop_all_nan_rows:
        merged_df = merged_df.dropna(subset=label_cols, how="all").reset_index(drop=True)

    def row_to_labels(row) -> List[float]:
        vals = row[label_cols].astype("float32").values
        return vals.tolist()

    merged_df["labels"] = merged_df.apply(row_to_labels, axis=1)
    df_for_hf = merged_df[
        ["report_txt", "image_path", "labels", "dicom_id", "subject_id", "study_id"]
    ].rename(columns={"report_txt": "text", "image_path": "image"}).copy()

    features = Features(
        {
            "text": Value("string"),
            "image": DatasetsImage(),
            "labels": Sequence(Value("float32")),
            "subject_id": Value("int64"),
            "study_id": Value("int64"),
            "dicom_id": Value("string"),
        }
    )

    hf_ds = Dataset.from_pandas(df_for_hf, features=features, preserve_index=False)
    num_labels = len(label_cols)
    class_names = label_cols

    print(f"[Dataset] Built: {len(hf_ds)} examples | num_labels={num_labels}")
    return hf_ds, num_labels, class_names


DATASET_CACHE_FORMAT_VERSION = 1


def _dataset_source_fingerprint(path: str) -> Dict[str, Any]:
    """Return inexpensive file metadata for cache invalidation."""
    absolute_path = os.path.abspath(path)
    stat = os.stat(absolute_path)
    return {"path": absolute_path, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _dataset_cache_spec(
    top_level_buckets, best_img_index_root, chexpert_csv, split_csv_path,
    report_root, image_root, cache_tag,
):
    source_files = [chexpert_csv, split_csv_path]
    for bucket in top_level_buckets:
        source_files.extend([
            os.path.join(best_img_index_root, bucket, f"{bucket}_best_img_final.csv"),
            os.path.join(report_root, bucket, AUDIT_CSV_NAME),
        ])
    return {
        "cache_format_version": DATASET_CACHE_FORMAT_VERSION,
        "cache_tag": str(cache_tag),
        "top_level_buckets": list(top_level_buckets),
        "xrv_root": os.path.abspath(best_img_index_root),
        "image_root": os.path.abspath(image_root) if image_root else None,
        "report_root": os.path.abspath(report_root),
        "drop_all_nan_rows": True,
        "strict_split_matching": True,
        "keep_ids": True,
        "sources": [_dataset_source_fingerprint(path) for path in source_files],
    }


def load_or_build_cached_mimic_splits(
    *, top_level_buckets, best_img_index_root, chexpert_csv, split_csv_path,
    report_root, image_root=None, cache_root, cache_tag="v1", rebuild_cache=False,
) -> Tuple[DatasetDict, int, List[str], str]:
    """Load cached official splits, or build and atomically save them once."""
    spec = _dataset_cache_spec(
        top_level_buckets, best_img_index_root, chexpert_csv, split_csv_path,
        report_root, image_root, cache_tag,
    )
    spec_json = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    cache_hash = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()[:16]
    bucket_name = "-".join(top_level_buckets)
    cache_dir = os.path.join(
        os.path.abspath(cache_root), f"mimic_chexpert_{bucket_name}_{cache_hash}"
    )
    metadata_path = os.path.join(cache_dir, "cache_metadata.json")

    if rebuild_cache and os.path.isdir(cache_dir):
        print(f"[Dataset cache] Rebuilding; removing old cache: {cache_dir}")
        shutil.rmtree(cache_dir)

    if os.path.isfile(metadata_path):
        print(f"[Dataset cache] Loading cached splits from: {cache_dir}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        if metadata.get("spec") != spec:
            raise RuntimeError(
                "Dataset cache metadata does not match its cache key. Run with --rebuild_dataset_cache."
            )
        cached_splits = load_from_disk(cache_dir)
        required_splits = {"train", "validation", "test"}
        if not isinstance(cached_splits, DatasetDict) or set(cached_splits.keys()) != required_splits:
            raise RuntimeError(
                f"Invalid cached dataset at {cache_dir}. Run with --rebuild_dataset_cache."
            )
        class_names = list(metadata["class_names"])
        num_labels = int(metadata["num_labels"])
        print(
            "[Dataset cache] Loaded: "
            f"train={len(cached_splits['train'])} "
            f"val={len(cached_splits['validation'])} "
            f"test={len(cached_splits['test'])}"
        )
        return cached_splits, num_labels, class_names, cache_dir

    if os.path.exists(cache_dir):
        raise RuntimeError(
            f"Incomplete dataset cache found at {cache_dir}. Run with --rebuild_dataset_cache to replace it."
        )

    print(f"[Dataset cache] No cache found. Building dataset for: {cache_dir}")
    ds, num_labels, class_names = build_mimic_chexpert_dataset(
        top_level_buckets=top_level_buckets, best_img_index_root=best_img_index_root,
        chexpert_csv=chexpert_csv, report_root=report_root, image_root=image_root,
        drop_all_nan_rows=True,
    )
    splits = split_mimic_dataset_official_by_ids(
        ds=ds, split_csv_path=split_csv_path, strict=True, keep_ids=True,
    )
    os.makedirs(cache_root, exist_ok=True)
    temporary_cache_dir = f"{cache_dir}.building-{os.getpid()}-{time.time_ns()}"
    try:
        print(f"[Dataset cache] Saving splits to: {cache_dir}")
        splits.save_to_disk(temporary_cache_dir)
        metadata = {
            "spec": spec,
            "num_labels": int(num_labels),
            "class_names": list(class_names),
            "split_sizes": {name: len(dataset) for name, dataset in splits.items()},
        }
        with open(os.path.join(temporary_cache_dir, "cache_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        os.rename(temporary_cache_dir, cache_dir)
    except Exception:
        if os.path.isdir(temporary_cache_dir):
            shutil.rmtree(temporary_cache_dir)
        raise
    print("[Dataset cache] Saved successfully.")
    return splits, num_labels, class_names, cache_dir
