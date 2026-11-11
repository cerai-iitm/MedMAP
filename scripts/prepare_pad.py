#!/usr/bin/env python3
"""Download/validate PAD-UFES-20 and optionally cache EfficientNet features."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from scripts._bootstrap import setup_project_paths

setup_project_paths()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--extract-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the frozen EfficientNet-B3 feature cache (default: enabled).",
    )
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value.")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    data_root = Path(args.data_root)
    image_root = data_root / "all_images"
    metadata_path = data_root / "metadata.csv"
    cache_path = Path(args.cache_path)

    from medmap.pad.preprocessing import download_and_extract

    if args.download:
        download_and_extract(str(data_root), str(image_root))

    if not metadata_path.is_file():
        parser.error(f"PAD metadata.csv is missing: {metadata_path}")
    if not image_root.is_dir():
        parser.error(f"PAD all_images directory is missing: {image_root}")

    import pandas as pd

    metadata = pd.read_csv(metadata_path, usecols=["img_id"])
    image_ids = metadata["img_id"].astype(str).tolist()
    missing = [image_id for image_id in image_ids if not (image_root / image_id).is_file()]
    if missing:
        examples = ", ".join(missing[:5])
        parser.error(
            f"PAD download is incomplete: {len(missing)} metadata images are missing "
            f"from {image_root}. Examples: {examples}"
        )
    print(f"PAD data validation passed: {len(image_ids)} metadata rows and matching images.")

    if not args.extract_features:
        return

    import torch
    from medmap.pad.features import load_or_build_raw_features

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = load_or_build_raw_features(
        str(cache_path),
        str(metadata_path),
        str(image_root),
        str(data_root),
        False,
        device,
    )
    image_features = result[0]
    print(
        f"PAD preprocessing complete: features={image_features.shape}, "
        f"cache={cache_path}, device={device}"
    )


if __name__ == "__main__":
    main()
