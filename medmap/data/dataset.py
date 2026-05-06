import os
import zipfile
import pandas as pd
from typing import List, Tuple
from datasets import Dataset, DatasetDict, Features, Value, Sequence, ClassLabel, Image as DatasetsImage
from ..utils.core import _safe_read_text
import numpy as np

IMAGE_TOKEN = "<image>"

def read_imagenet_train_split(root_or_zip: str, train_txt_rel: str) -> pd.DataFrame:
    if os.path.isdir(root_or_zip):
        txt_path = os.path.join(root_or_zip, train_txt_rel)
        with open(txt_path, "r") as f:
            rows = [l.strip().split() for l in f if l.strip()]
        return pd.DataFrame(rows, columns=["wnid/filename", "label"])

    with zipfile.ZipFile(root_or_zip) as z, z.open(train_txt_rel) as f:
        rows = [l.decode().strip().split() for l in f if l.strip()]
    return pd.DataFrame(rows, columns=["wnid/filename", "label"])

def build_imagenet_dataset(captions_json: str, root_or_zip: str, train_txt_rel: str, use_tiny: bool = True):
    caps = pd.read_json(captions_json)
    caps["wnid/filename"] = caps["wnid"] + "/" + caps["filename"].str.replace(".JPEG", "", regex=False)

    train_df = read_imagenet_train_split(root_or_zip, train_txt_rel)

    df = caps.merge(train_df, on="wnid/filename")
    df["tags_flat"] = df["tags"].apply(", ".join)
    df["prompt"] = IMAGE_TOKEN + " " + df["title"] + " " + df["tags_flat"] + " " + df["description"]

    is_fs = os.path.isdir(root_or_zip)

    def _img_path(r):
        if is_fs:
            return os.path.join(root_or_zip, "ILSVRC", "Data", "CLS-LOC", "train", r["wnid"], r["filename"])
        else:
            return f"zip://ILSVRC/Data/CLS-LOC/train/{r['wnid']}/{r['filename']}::{root_or_zip}"

    df["image_path"] = df.apply(_img_path, axis=1)

    if use_tiny:
        tiny_df = pd.read_csv("/data/common/data/tiny-imagenet-200/wnids.txt", names=['wnid'])
        keep = set(tiny_df['wnid'].astype(str).str.strip().str.lower().dropna().unique())
        df = df[df['wnid'].astype(str).str.strip().str.lower().isin(keep)].copy()
        df = df.reset_index(drop=True)

    wnids = df["wnid"].unique().tolist()
    id_map = {w: i for i, w in enumerate(wnids)}
    df["class_idx"] = df["wnid"].map(id_map).astype("int64")
    num_classes = len(wnids)

    print("Total records: ", len(df))
    print("Total classes: ", num_classes)

    features = Features({
        "prompt": Value("string"), 
        "class_idx": ClassLabel(num_classes=num_classes, names=wnids), 
        "image_path": DatasetsImage()
    })
    
    hf_ds = Dataset.from_pandas(df[["prompt", "class_idx", "image_path"]], features=features, preserve_index=False)
    return hf_ds, num_classes, wnids


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
    val_ds   = ds2.filter(lambda x: x["__split"] == "validate")
    test_ds  = ds2.filter(lambda x: x["__split"] == "test")

    # 7) Cleanup helper and optionally drop id columns
    drop_cols = ["__split"]
    if not keep_ids:
        drop_cols += ["dicom_id", "subject_id", "study_id"]

    train_ds = train_ds.remove_columns(drop_cols)
    val_ds   = val_ds.remove_columns(drop_cols)
    test_ds  = test_ds.remove_columns(drop_cols)

    return DatasetDict(train=train_ds, validation=val_ds, test=test_ds)



def build_mimic_chexpert_dataset(
    top_level_buckets: List[str],
    mimic_best_img_dump: str,
    chexpert_csv: str,
    report_root: str,
    drop_all_nan_rows: bool = True,
) -> Tuple[Dataset, int, List[str]]:
    """
    Builds an HF Dataset with columns:
      - text: raw report text (may be empty if report missing)
      - image: datasets.Image (path stored; decoded on-the-fly)
      - labels: list[float32] with {0.0, 1.0, NaN} (NaNs ignored in loss)

    Notes:
    - CheXpert convention: -1 = uncertain -> convert to NaN so it is ignored in loss.
    """
    best_df_concat = None
    for top_level_bucket in top_level_buckets:
        best_csv_path = os.path.join(f"medmap/mimic_best_img_dump/{top_level_bucket}_best_image_per_study_all_heuristic.csv")
        # 1) Base CSV with best image per study
        best_df = pd.read_csv(best_csv_path, usecols=['image_path', 'subject_id', 'study_id', 'dicom_id'])

        best_df['subject_id'] = best_df['subject_id'].astype(int)
        best_df['study_id'] = best_df['study_id'].astype(int)
        best_df['dicom_id'] = best_df['dicom_id'].astype(str)
        print(f"Loaded {len(best_df)} entries from {best_csv_path}")

        # Build report paths:
        # <reports_root_p10>/p<subject_id>/s<study_id>.txt
        best_df["report_path"] = (
            f"{report_root}/{top_level_bucket}"
            + "/p"
            + best_df['subject_id'].astype(str)
            + "/s"
            + best_df['study_id'].astype(str)
            + ".txt"
        )
        best_df["report_txt"] = best_df["report_path"].apply(_safe_read_text)

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
    df_for_hf = merged_df[["report_txt", "image_path", "labels", "dicom_id", "subject_id", "study_id"]].rename(columns={"report_txt": "text", "image_path": "image"}).copy()

    # Take a subset for faster testing
    # Fix: for tiny debug size, set eval_steps smaller than steps/epoch, e.g. 5 or 10.
    # Random sampling can produce empty val/test splits; Fix: sample after splitting (or sample per split).
    # Many metrics become NaN-only on a 100-sample subset, and W&B won’t plot them
    # Fix: if you want those plots during tiny tests, you need stratified sampling (or accept that AUC/AUPRC won’t be meaningful on 100 rows).
    df_for_hf = df_for_hf.sample(n=200, random_state=42).reset_index(drop=True)

    features = Features(
        {
            "text": Value("string"),
            "image": DatasetsImage(),
            "labels": Sequence(Value("float32")),
            "subject_id": Value("int64"),
            "study_id": Value("int64"),
            "dicom_id": Value("string")
        }
    )

    hf_ds = Dataset.from_pandas(df_for_hf, features=features, preserve_index=False)
    num_labels = len(label_cols)
    class_names = label_cols

    print(f"[Dataset] Built: {len(hf_ds)} examples | num_labels={num_labels}")
    return hf_ds, num_labels, class_names