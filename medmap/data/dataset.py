import os
import zipfile
import pandas as pd
from datasets import Dataset, Features, Value, ClassLabel, Image as DatasetsImage

IMAGE_TOKEN = "<image>"

def read_train_split(root_or_zip: str, train_txt_rel: str) -> pd.DataFrame:
    if os.path.isdir(root_or_zip):
        txt_path = os.path.join(root_or_zip, train_txt_rel)
        with open(txt_path, "r") as f:
            rows = [l.strip().split() for l in f if l.strip()]
        return pd.DataFrame(rows, columns=["wnid/filename", "label"])

    with zipfile.ZipFile(root_or_zip) as z, z.open(train_txt_rel) as f:
        rows = [l.decode().strip().split() for l in f if l.strip()]
    return pd.DataFrame(rows, columns=["wnid/filename", "label"])

def build_dataset(captions_json: str, root_or_zip: str, train_txt_rel: str, use_tiny: bool = True):
    caps = pd.read_json(captions_json)
    caps["wnid/filename"] = caps["wnid"] + "/" + caps["filename"].str.replace(".JPEG", "", regex=False)

    train_df = read_train_split(root_or_zip, train_txt_rel)

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