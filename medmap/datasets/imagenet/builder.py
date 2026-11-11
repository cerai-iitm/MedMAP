"""ImageNet-Captions / Tiny ImageNet dataset builder for SmolVLM.

Joins the ImageNet-Captions JSON (title/tags/description per image) with the
ILSVRC train split listing, builds ``<image> <title> <tags> <description>``
prompts, and (by default) filters to the 200 Tiny ImageNet classes using the
``wnids.txt`` file that ships with tiny-imagenet-200.

Copied as-is from the original MedMAP implementation, with one change: the
Tiny ImageNet ``wnids.txt`` location is configurable via the
``TINY_IMAGENET_WNIDS`` environment variable (or the ``tiny_wnids_path``
argument) instead of a machine-specific hardcoded path.
"""

import os
import zipfile
from typing import Optional

import pandas as pd
from datasets import ClassLabel, Dataset, Features, Value
from datasets import Image as DatasetsImage

IMAGE_TOKEN = "<image>"


def _resolve_tiny_wnids_path(tiny_wnids_path: Optional[str]) -> str:
    path = tiny_wnids_path or os.environ.get("TINY_IMAGENET_WNIDS", "")
    if not path:
        raise RuntimeError(
            "Tiny ImageNet class list not configured. Set TINY_IMAGENET_WNIDS to "
            "the wnids.txt of tiny-imagenet-200 (see docs/imagenet.md), e.g.\n"
            "    export TINY_IMAGENET_WNIDS=/data/tiny-imagenet-200/wnids.txt"
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"TINY_IMAGENET_WNIDS does not exist: {path}")
    return path


def read_imagenet_train_split(root_or_zip: str, train_txt_rel: str) -> pd.DataFrame:
    if os.path.isdir(root_or_zip):
        txt_path = os.path.join(root_or_zip, train_txt_rel)
        with open(txt_path, "r") as f:
            rows = [l.strip().split() for l in f if l.strip()]
        return pd.DataFrame(rows, columns=["wnid/filename", "label"])

    with zipfile.ZipFile(root_or_zip) as z, z.open(train_txt_rel) as f:
        rows = [l.decode().strip().split() for l in f if l.strip()]
    return pd.DataFrame(rows, columns=["wnid/filename", "label"])


def build_imagenet_dataset(
    captions_json: str,
    root_or_zip: str,
    train_txt_rel: str,
    use_tiny: bool = True,
    tiny_wnids_path: Optional[str] = None,
    validate_images: bool = False,
):
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
        wnids_path = _resolve_tiny_wnids_path(tiny_wnids_path)
        tiny_df = pd.read_csv(wnids_path, names=['wnid'])
        keep = set(tiny_df['wnid'].astype(str).str.strip().str.lower().dropna().unique())
        df = df[df['wnid'].astype(str).str.strip().str.lower().isin(keep)].copy()
        df = df.reset_index(drop=True)

    if validate_images:
        if is_fs:
            missing_images = [path for path in df["image_path"] if not os.path.isfile(path)]
        else:
            with zipfile.ZipFile(root_or_zip) as archive:
                members = set(archive.namelist())
            expected_members = [
                f"ILSVRC/Data/CLS-LOC/train/{row.wnid}/{row.filename}"
                for row in df[["wnid", "filename"]].itertuples(index=False)
            ]
            missing_images = [path for path in expected_members if path not in members]
        if missing_images:
            examples = ", ".join(str(path) for path in missing_images[:5])
            raise FileNotFoundError(
                f"ILSVRC source is incomplete: {len(missing_images)} joined images "
                f"are missing. Examples: {examples}"
            )

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
