#!/usr/bin/env python3
"""Select exactly one MIMIC-CXR JPG per study using DICOM metadata.

The metadata-only selection hierarchy is:

    PA > AP > LATERAL > OTHER_OR_UNKNOWN

``ViewPosition`` is the primary projection field.  When it is missing or does
not contain a recognized standard projection, ``ViewCodeSequence_CodeMeaning``
is used as a fallback.  If multiple images have the best available projection,
the lexicographically smallest ``dicom_id`` is selected.  The selector never
uses report text, clinical labels, image content, or model predictions.

Examples (via the repository CLI wrapper):

    python -m scripts.select_mimic_images p10 \
        --dataset-root /data/mimic/physionet.org/files/mimic-cxr-jpg/2.1.0 \
        --output-root  /data/mimic/best_img_index
    python -m scripts.select_mimic_images p10 p11 p12 --dataset-root ... --output-root ...

Shard pXX is written to ``<output-root>/pXX/pXX_best_img_final.csv`` with the
columns (image_path, subject_id, study_id, dicom_id), exactly one row per
study. This is the index consumed by the MIMIC-CXR dataset builder.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


OUTPUT_COLUMNS = ("image_path", "subject_id", "study_id", "dicom_id")
REQUIRED_METADATA_COLUMNS = (
    "dicom_id",
    "subject_id",
    "study_id",
    "ViewPosition",
    "ViewCodeSequence_CodeMeaning",
)

SHARD_PATTERN = re.compile(r"p1[0-9]")

PA = "PA"
AP = "AP"
LATERAL = "LATERAL"
OTHER_OR_UNKNOWN = "OTHER_OR_UNKNOWN"

PROJECTION_PRIORITY = {
    PA: 0,
    AP: 1,
    LATERAL: 2,
    OTHER_OR_UNKNOWN: 3,
}

LATERAL_VIEW_POSITIONS = {"LATERAL", "LL", "XTABLE LATERAL"}
LATERAL_VIEW_CODES = {"LATERAL", "LEFT LATERAL"}


@dataclass(frozen=True)
class Candidate:
    """One eligible image and its deterministic selection rank."""

    image_path: Path
    subject_id: str
    study_id: str
    dicom_id: str
    projection: str

    @property
    def rank(self) -> tuple[int, str]:
        return (PROJECTION_PRIORITY[self.projection], self.dicom_id)


def normalize_metadata_value(value: str | None) -> str:
    """Normalize a possibly missing categorical DICOM metadata value."""

    if value is None:
        return ""
    return value.strip().upper()


def classify_projection(view_position: str | None, view_code: str | None) -> str:
    """Map the two DICOM-derived fields to one projection category.

    A recognized ``ViewPosition`` takes precedence.  The view code is only
    consulted when ``ViewPosition`` is missing or unrecognized.  Values such as
    ERECT and RECUMBENT are deliberately not treated as projections.
    """

    normalized_position = normalize_metadata_value(view_position)
    normalized_code = normalize_metadata_value(view_code)

    if normalized_position == PA:
        return PA
    if normalized_position == AP:
        return AP
    if normalized_position in LATERAL_VIEW_POSITIONS:
        return LATERAL

    if normalized_code == "POSTERO-ANTERIOR":
        return PA
    if normalized_code == "ANTERO-POSTERIOR":
        return AP
    if normalized_code in LATERAL_VIEW_CODES:
        return LATERAL

    return OTHER_OR_UNKNOWN


def shard_for_subject(subject_id: str) -> str:
    """Return the MIMIC top-level shard for a subject identifier."""

    subject_id = subject_id.strip()
    if len(subject_id) < 2 or not subject_id.isdigit():
        raise ValueError(f"Invalid subject_id in metadata: {subject_id!r}")
    return f"p{subject_id[:2]}"


def image_path_for_row(dataset_root: Path, shard: str, row: dict[str, str]) -> Path:
    """Construct the absolute JPG path represented by a metadata row."""

    subject_id = row["subject_id"].strip()
    study_id = row["study_id"].strip()
    dicom_id = row["dicom_id"].strip()
    return (
        dataset_root
        / "files"
        / shard
        / f"p{subject_id}"
        / f"s{study_id}"
        / f"{dicom_id}.jpg"
    )


def normalize_shards(raw_shards: Sequence[str]) -> list[str]:
    """Accept space-separated and/or comma-separated shard arguments."""

    normalized: list[str] = []
    for raw_value in raw_shards:
        for raw_shard in raw_value.split(","):
            shard = raw_shard.strip().lower()
            if not shard:
                continue
            if shard.isdigit():
                shard = f"p{shard}"
            if not SHARD_PATTERN.fullmatch(shard):
                raise ValueError(
                    f"Invalid shard {raw_shard!r}; expected p10 through p19."
                )
            if shard not in normalized:
                normalized.append(shard)

    if not normalized:
        raise ValueError("At least one shard must be provided.")
    return normalized


def validate_metadata_header(fieldnames: Iterable[str] | None) -> None:
    """Raise a clear error when the metadata schema is not as expected."""

    available = set(fieldnames or ())
    missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in available]
    if missing:
        raise ValueError(
            "Metadata CSV is missing required columns: " + ", ".join(missing)
        )


def select_best_images(
    metadata_csv: Path,
    dataset_root: Path,
    shards: Sequence[str],
    verify_files: bool,
) -> tuple[dict[str, list[Candidate]], dict[str, Counter[str]]]:
    """Read metadata once and select one ranked candidate for every study."""

    requested_shards = set(shards)
    best_by_shard: dict[str, dict[str, Candidate]] = {
        shard: {} for shard in shards
    }
    studies_seen: dict[str, set[str]] = {shard: set() for shard in shards}

    with metadata_csv.open("r", encoding="utf-8", newline="") as metadata_file:
        reader = csv.DictReader(metadata_file)
        validate_metadata_header(reader.fieldnames)

        for row in reader:
            subject_id = row["subject_id"].strip()
            shard = shard_for_subject(subject_id)
            if shard not in requested_shards:
                continue

            study_id = row["study_id"].strip()
            dicom_id = row["dicom_id"].strip()
            if not study_id or not dicom_id:
                raise ValueError(
                    "Encountered an empty study_id or dicom_id in metadata for "
                    f"subject_id={subject_id!r}."
                )

            studies_seen[shard].add(study_id)
            image_path = image_path_for_row(dataset_root, shard, row)
            if verify_files and not image_path.is_file():
                continue

            candidate = Candidate(
                image_path=image_path,
                subject_id=subject_id,
                study_id=study_id,
                dicom_id=dicom_id,
                projection=classify_projection(
                    row.get("ViewPosition"),
                    row.get("ViewCodeSequence_CodeMeaning"),
                ),
            )

            current_best = best_by_shard[shard].get(study_id)
            if current_best is None or candidate.rank < current_best.rank:
                best_by_shard[shard][study_id] = candidate

    selected: dict[str, list[Candidate]] = {}
    tier_counts: dict[str, Counter[str]] = {}

    for shard in shards:
        missing_studies = studies_seen[shard] - best_by_shard[shard].keys()
        if missing_studies:
            examples = ", ".join(sorted(missing_studies)[:10])
            raise RuntimeError(
                f"{shard}: {len(missing_studies)} studies have metadata but no "
                "available JPG file. No CSV was written because silently dropping "
                f"studies would violate the one-image-per-study invariant. Examples: "
                f"{examples}"
            )

        candidates = list(best_by_shard[shard].values())
        candidates.sort(
            key=lambda candidate: (
                int(candidate.subject_id),
                int(candidate.study_id),
                candidate.dicom_id,
            )
        )

        if len(candidates) != len(studies_seen[shard]):
            raise AssertionError(
                f"{shard}: internal invariant failure: selected {len(candidates)} "
                f"images for {len(studies_seen[shard])} studies."
            )

        selected[shard] = candidates
        tier_counts[shard] = Counter(
            candidate.projection for candidate in candidates
        )

    return selected, tier_counts


def write_selection_csv(
    output_path: Path,
    candidates: Sequence[Candidate],
    *,
    image_root: Path | None = None,
) -> None:
    """Atomically write the image index.

    When ``image_root`` is provided, image paths are stored relative to that
    root (for example ``files/p10/.../image.jpg``). Relative indexes are
    portable across machines and are resolved by the dataset builder through
    its explicit ``image_root`` argument. Omitting ``image_root`` preserves
    the legacy absolute-path behavior for API callers.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for candidate in candidates:
                image_path = candidate.image_path
                if image_root is not None:
                    try:
                        image_path = image_path.resolve().relative_to(image_root.resolve())
                    except ValueError as error:
                        raise ValueError(
                            f"Selected image is outside --dataset-root: {image_path}"
                        ) from error
                writer.writerow(
                    {
                        "image_path": image_path.as_posix(),
                        "subject_id": candidate.subject_id,
                        "study_id": candidate.study_id,
                        "dicom_id": candidate.dicom_id,
                    }
                )

        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select exactly one MIMIC-CXR image per study using the deterministic "
            "PA > AP > lateral > other/unknown metadata hierarchy."
        )
    )
    parser.add_argument(
        "shards",
        nargs="+",
        help="One or more shards (for example: p10 p11 p12).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "MIMIC-CXR-JPG version directory, e.g. "
            "<download>/physionet.org/files/mimic-cxr-jpg/2.1.0 "
            "(must contain files/ and the metadata CSV)."
        ),
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help=(
            "MIMIC-CXR-JPG metadata CSV. Default: "
            "<dataset-root>/mimic-cxr-2.0.0-metadata.csv[.gz]."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help=(
            "Best-image index root. Each shard is written to "
            "<output-root>/<shard>/<shard>_best_img_final.csv."
        ),
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help=(
            "Store absolute JPG paths in the generated CSV. By default paths are "
            "relative to --dataset-root and are portable across machines."
        ),
    )
    parser.add_argument(
        "--skip-file-verification",
        action="store_true",
        help=(
            "Do not check that each JPG exists. By default, missing JPGs are skipped "
            "and the next best available image is selected."
        ),
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    try:
        shards = normalize_shards(args.shards)
    except ValueError as error:
        parser.error(str(error))

    if not args.dataset_root.is_dir():
        parser.error(f"Dataset root does not exist: {args.dataset_root}")
    if args.metadata_csv is None:
        args.metadata_csv = args.dataset_root / "mimic-cxr-2.0.0-metadata.csv"
        if not args.metadata_csv.is_file() and args.metadata_csv.with_suffix(
            ".csv.gz"
        ).is_file():
            parser.error(
                f"Found {args.metadata_csv}.gz but not the uncompressed CSV. "
                "Run `gunzip -k` on it first (the selector streams plain CSV)."
            )
    if not args.metadata_csv.is_file():
        parser.error(f"Metadata CSV does not exist: {args.metadata_csv}")

    selected, tier_counts = select_best_images(
        metadata_csv=args.metadata_csv,
        dataset_root=args.dataset_root,
        shards=shards,
        verify_files=not args.skip_file_verification,
    )

    for shard in shards:
        output_path = (
            args.output_root
            / shard
            / f"{shard}_best_img_final.csv"
        )
        write_selection_csv(
            output_path,
            selected[shard],
            image_root=None if args.absolute_paths else args.dataset_root,
        )

        counts = tier_counts[shard]
        print(
            f"{shard}: wrote {len(selected[shard]):,} images "
            f"(PA={counts[PA]:,}, AP={counts[AP]:,}, "
            f"LATERAL={counts[LATERAL]:,}, "
            f"OTHER_OR_UNKNOWN={counts[OTHER_OR_UNKNOWN]:,}) to {output_path}"
        )


if __name__ == "__main__":
    main()
