import os
import zipfile
import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel, Image as DatasetsImage
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

def split_mimic_dataset_official_by_ids(ds: Dataset, split_csv_path: str, strict: bool = True, keep_ids: bool = True) -> DatasetDict:
    split_df = pd.read_csv(split_csv_path, usecols=["dicom_id", "subject_id", "study_id", "split"])
    split_df["dicom_id"] = split_df["dicom_id"].astype(str)
    split_df["subject_id"] = split_df["subject_id"].astype("int64")
    split_df["study_id"] = split_df["study_id"].astype("int64")
    
    split_map = {(r["dicom_id"], int(r["subject_id"]), int(r["study_id"])): r["split"] for _, r in split_df.iterrows()}
    
    def add_split(ex):
        key = (str(ex["dicom_id"]), int(ex["subject_id"]), int(ex["study_id"]))
        return {"__split": split_map.get(key, None)}

    ds2 = ds.map(add_split, desc="Assign official split via triple-key")
    if strict and len(ds2.filter(lambda x: x["__split"] is None)) > 0:
        raise ValueError("Missing rows in split match.")
        
    ds2 = ds2.filter(lambda x: x["__split"] is not None)
    
    splits = {
        "train": ds2.filter(lambda x: x["__split"] == "train"),
        "validation": ds2.filter(lambda x: x["__split"] == "validate"),
        "test": ds2.filter(lambda x: x["__split"] == "test")
    }
    
    drop_cols = ["__split"] if keep_ids else ["__split", "dicom_id", "subject_id", "study_id"]
    return DatasetDict({k: v.remove_columns(drop_cols) for k, v in splits.items()})


def build_mimic_chexpert_dataset(top_level_buckets, xrv_root, chexpert_csv, report_root, drop_all_nan_rows=True):
    best_df_concat = None
    for bucket in top_level_buckets:
        best_csv = os.path.join(xrv_root, bucket, "best_img", f"{bucket}_best_image_per_study_all_heuristic.csv")
        df = pd.read_csv(best_csv, usecols=['image_path', 'subject_id', 'study_id', 'dicom_id'])
        df["report_path"] = f"{report_root}/{bucket}/p" + df['subject_id'].astype(str) + "/s" + df['study_id'].astype(str) + "_masked.txt"
        
        # Helper: _safe_read_text from MedMAP.utils.core can be imported here, or defined inline:
        df["report_txt"] = df["report_path"].apply(lambda p: open(p).read().strip() if os.path.exists(p) else "")
        best_df_concat = pd.concat([best_df_concat, df], ignore_index=True) if best_df_concat is not None else df

    chx_df = pd.read_csv(chexpert_csv)
    chx_df = chx_df[chx_df["study_id"].isin(best_df_concat["study_id"].unique())]
    
    id_cols = ["subject_id", "study_id"]
    label_cols = [c for c in chx_df.columns if c not in id_cols and c not in ["Support Devices", "No Finding"]]
    
    merged = best_df_concat.merge(chx_df[id_cols + label_cols], on=id_cols, how="left", validate="one_to_one")
    merged[label_cols] = merged[label_cols].replace(-1, np.nan)
    
    if drop_all_nan_rows:
        merged = merged.dropna(subset=label_cols, how="all").reset_index(drop=True)

    merged["labels"] = merged.apply(lambda r: r[label_cols].astype("float32").tolist(), axis=1)
    df_for_hf = merged[["report_txt", "image_path", "labels", "dicom_id", "subject_id", "study_id"]].rename(columns={"report_txt": "text", "image_path": "image"})
    
    from datasets import Features, Value, Sequence, Image as DatasetsImage
    features = Features({
        "text": Value("string"), "image": DatasetsImage(), "labels": Sequence(Value("float32")),
        "subject_id": Value("int64"), "study_id": Value("int64"), "dicom_id": Value("string")
    })
    
    return Dataset.from_pandas(df_for_hf, features=features, preserve_index=False), len(label_cols), label_cols