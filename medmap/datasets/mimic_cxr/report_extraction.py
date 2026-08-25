#!/usr/bin/env python3
"""Leakage-filtered MIMIC-CXR report extraction.

Removes the label-source sections (Findings / Impression) with the official
MIT-LCP section parser, then masks official CheXpert / NegBio label mentions
in the remaining text. Each original report ``s<study_id>.txt`` is preserved
and a sibling ``s<study_id>_mmp.txt`` is written next to it, together with a
per-shard audit CSV listing reports that have no usable Findings/Impression
section (those studies are excluded downstream).

The section parser and phrase resources are vendored, pinned copies of the
official repositories used to create MIMIC-CXR-JPG labels (see
``vendor/SOURCES.md``). Original reports are never modified. Only canonical
``s<study_id>.txt`` files are inputs, so generated files can never be
recursively reprocessed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
import time
from copy import copy
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import yaml

from medmap.datasets.mimic_cxr.vendor.mimic_code import section_parser


PROJECT_DIR = Path(__file__).resolve().parent

# Suffix appended to every generated leakage-filtered report. The pipeline
# writes ``s<study_id>{REPORT_SUFFIX}.txt`` beside each original report and a
# per-shard audit CSV named ``reports_without_findings_or_impression{REPORT_SUFFIX}.csv``.
REPORT_SUFFIX = "_mmp"
AUDIT_CSV_NAME = f"reports_without_findings_or_impression{REPORT_SUFFIX}.csv"

CHEXPERT_MENTION_DIR = PROJECT_DIR / "vendor/chexpert_labeler/phrases/mention"
CHEXPERT_UNMENTION_DIR = PROJECT_DIR / "vendor/chexpert_labeler/phrases/unmention"
NEGBIO_PHRASES = PROJECT_DIR / "vendor/negbio/patterns/chexpert_phrases.yml"
CANONICAL_REPORT_RE = re.compile(r"^s(?P<study_id>\d+)\.txt$")
SHARD_RE = re.compile(r"^p\d{2}$")
OFFICIAL_HEADER_RE = re.compile(r"\n ([A-Z ()/,-]+):\s", re.DOTALL)
WORD_CHAR_RE = re.compile(r"[A-Za-z0-9_-]")


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    sources: tuple[str, ...] = ()


@dataclass
class RunStats:
    shard: str
    discovered: int = 0
    processed: int = 0
    written: int = 0
    skipped_existing: int = 0
    without_findings_or_impression: int = 0
    section_characters_removed: int = 0
    masked_spans: int = 0
    errors: int = 0
    dry_run: bool = False


class ProgressReporter:
    """Lightweight, dependency-free terminal progress with rate and ETA."""

    def __init__(self, version: str, shard: str, total: int, every: int = 100):
        self.version = version
        self.shard = shard
        self.total = total
        self.every = max(1, every)
        self.started = time.monotonic()
        print(f"[{version}][{shard}] Starting {total:,} canonical reports", flush=True)

    @staticmethod
    def _duration(seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def update(self, completed: int) -> None:
        if completed != self.total and completed != 1 and completed % self.every:
            return
        elapsed = max(time.monotonic() - self.started, 1e-9)
        rate = completed / elapsed
        remaining = max(self.total - completed, 0)
        eta = remaining / rate if rate else 0
        percent = 100.0 if self.total == 0 else completed * 100.0 / self.total
        print(
            f"[{self.version}][{self.shard}] {completed:,}/{self.total:,} "
            f"({percent:5.1f}%) | {rate:,.1f} reports/s | "
            f"elapsed {self._duration(elapsed)} | ETA {self._duration(eta)}",
            flush=True,
        )


def _official_header_sections(text: str) -> list[tuple[int, int, int, str]]:
    """Return header/content spans using section_parser.section_text search logic.

    Tuples are ``(header_start, content_start, section_end, normalized_name)``.
    The parser deliberately skips the first newline after a header before
    searching for another header; that behavior is retained verbatim here.
    """
    found: list[tuple[int, int, int, str]] = []
    match = OFFICIAL_HEADER_RE.search(text, 0)
    while match:
        content_start = match.end()
        skip = text[content_start:].find("\n")
        if skip == -1:
            skip = 0
        next_match = OFFICIAL_HEADER_RE.search(text, content_start + skip)
        section_end = len(text) if next_match is None else next_match.start()
        normalized = section_parser.normalize_section_names([match.group(1)])[0]
        # Preserve the newline immediately before each header.
        found.append((match.start() + 1, content_start, section_end + 1, normalized))
        match = next_match
    if found:
        # The last section ends at EOF; +1 above is only meaningful for a next header.
        hs, cs, _, name = found[-1]
        found[-1] = (hs, cs, len(text), name)
    return found


def _section_removal_spans(text: str, study_stem: str) -> tuple[list[Span], bool]:
    """Apply official parsing plus the official MIMIC manual overrides."""
    headers = _official_header_sections(text)
    custom_names, custom_indices = _official_custom_rules()
    spans: list[Span] = []

    # Mirror section_text's removal of empty findings/impression sections when
    # deciding whether an explicit, usable section exists.
    usable_explicit = False
    for header_start, content_start, section_end, name in headers:
        if name in {"findings", "impression"}:
            content_end = section_end
            if content_end < len(text):
                content_end -= 1
            if text[content_start:content_end].strip():
                usable_explicit = True
            spans.append(Span(header_start, section_end, (f"section:{name}",)))

    # create_section_files.py gives these overrides priority over parsing.
    if study_stem in custom_indices:
        start, end = custom_indices[study_stem]
        if end > start:
            spans.append(Span(start, end, ("mimic_manual_index",)))
            usable_explicit = True
        else:
            usable_explicit = False
    elif study_stem in custom_names:
        target_name = custom_names[study_stem]
        candidates = [h for h in headers if h[3] == target_name]
        if candidates:
            header_start, _, section_end, _ = candidates[-1]
            spans.append(Span(header_start, section_end, ("mimic_manual_name",)))
            usable_explicit = True

    return merge_spans(spans), not usable_explicit


def merge_spans(spans: Iterable[Span]) -> list[Span]:
    ordered = sorted((s for s in spans if s.end > s.start), key=lambda s: (s.start, s.end))
    if not ordered:
        return []
    merged: list[Span] = [ordered[0]]
    for span in ordered[1:]:
        prior = merged[-1]
        if span.start <= prior.end:
            merged[-1] = Span(
                prior.start,
                max(prior.end, span.end),
                tuple(sorted(set(prior.sources + span.sources))),
            )
        else:
            merged.append(span)
    return merged


def remove_spans(text: str, spans: Sequence[Span]) -> str:
    cursor = 0
    pieces: list[str] = []
    for span in spans:
        pieces.append(text[cursor:span.start])
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces)


@lru_cache(maxsize=1)
def _official_custom_rules() -> tuple[dict[str, str], dict[str, list[int]]]:
    return section_parser.custom_mimic_cxr_rules()


@lru_cache(maxsize=4)
def _load_chexpert_phrases(directory: Path) -> dict[str, list[str]]:
    phrases: dict[str, list[str]] = {}
    for path in sorted(directory.glob("*.txt")):
        observation = path.stem.replace("_", " ").title()
        # splitlines handles the official files that lack a final newline.
        phrases[observation] = [line.strip().replace("_", " ") for line in path.read_text().splitlines() if line.strip()]
    return phrases


def _add_positional_unmentions(
    mentions: dict[str, list[str]], unmentions: dict[str, list[str]], *, negbio: bool
) -> None:
    positional_left = ["over the", "overly the", "in the"]
    if negbio:
        positional_left += ["assessment of", "diameter of"]
    positional_right = ["", " superior", " left", " right"]
    positional = [a + b for a in positional_left for b in positional_right]
    cardio = mentions.get("Cardiomegaly", [])
    enlarged = mentions.get("Enlarged Cardiomediastinum", [])
    unmentions["Cardiomegaly"] = [
        p + " " + phrase.replace("the ", "")
        for p in positional
        for phrase in cardio
        if phrase not in ["cardiomegaly", "cardiac enlargement"]
    ]
    unmentions["Enlarged Cardiomediastinum"] = [
        p + " " + phrase for p in positional for phrase in enlarged
    ]


def _replace_mapped(
    chars: list[str], maps: list[tuple[int, int]], pattern: re.Pattern[str], replacement: str
) -> tuple[list[str], list[tuple[int, int]]]:
    source = "".join(chars)
    out_chars: list[str] = []
    out_maps: list[tuple[int, int]] = []
    cursor = 0
    for match in pattern.finditer(source):
        out_chars.extend(chars[cursor:match.start()])
        out_maps.extend(maps[cursor:match.start()])
        covered = maps[match.start():match.end()]
        mapped = (covered[0][0], covered[-1][1]) if covered else (0, 0)
        out_chars.extend(replacement)
        out_maps.extend([mapped] * len(replacement))
        cursor = match.end()
    out_chars.extend(chars[cursor:])
    out_maps.extend(maps[cursor:])
    return out_chars, out_maps


def chexpert_clean_with_mapping(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Reproduce official Loader.clean while retaining original offsets."""
    chars = list(text.lower())
    maps = [(i, i + 1) for i in range(len(text))]
    chars, maps = _replace_mapped(chars, maps, re.compile("and/or"), "or")
    chars, maps = _replace_mapped(chars, maps, re.compile(r"(?<=[a-zA-Z])/(?=[a-zA-Z])"), " or ")
    chars, maps = _replace_mapped(chars, maps, re.compile(r"\.\."), ".")

    translated_chars: list[str] = []
    translated_maps: list[tuple[int, int]] = []
    for char, mapping in zip(chars, maps):
        translated_chars.append(char)
        translated_maps.append(mapping)
        if char in ".,":
            translated_chars.append(" ")
            translated_maps.append(mapping)
    chars, maps = translated_chars, translated_maps

    # Exact semantic equivalent of ' '.join(clean_report.split()), with maps.
    source = "".join(chars)
    compact_chars: list[str] = []
    compact_maps: list[tuple[int, int]] = []
    tokens = list(re.finditer(r"\S+", source))
    for index, token in enumerate(tokens):
        if index:
            gap_start, gap_end = tokens[index - 1].end(), token.start()
            compact_chars.append(" ")
            compact_maps.append((maps[gap_start][0], maps[gap_end - 1][1]))
        compact_chars.extend(chars[token.start():token.end()])
        compact_maps.extend(maps[token.start():token.end()])
    chars, maps = compact_chars, compact_maps
    chars, maps = _replace_mapped(chars, maps, re.compile(r"\.\s+\."), ".")
    return "".join(chars), maps


def _overlaps_any(start: int, end: int, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start < other_end and end > other_start for other_start, other_end in spans)


def chexpert_match_spans(text: str) -> list[Span]:
    """Use the official CheXpert Extractor phrase and unmention semantics."""
    mentions = _load_chexpert_phrases(CHEXPERT_MENTION_DIR)
    unmentions = _load_chexpert_phrases(CHEXPERT_UNMENTION_DIR)
    _add_positional_unmentions(mentions, unmentions, negbio=False)
    cleaned, mapping = chexpert_clean_with_mapping(text)
    found: list[Span] = []
    for observation, phrases in mentions.items():
        excluded: list[tuple[int, int]] = []
        for phrase in unmentions.get(observation, []):
            excluded.extend(match.span() for match in re.finditer(phrase, cleaned))
        for phrase in phrases:
            for match in re.finditer(phrase, cleaned):
                if _overlaps_any(match.start(), match.end(), excluded):
                    continue
                original_start = mapping[match.start()][0]
                original_end = mapping[match.end() - 1][1]
                found.append(Span(original_start, original_end, (f"chexpert:{observation}",)))
    return found


@lru_cache(maxsize=None)
def _negbio_compile(pattern: str) -> re.Pattern[str]:
    # Semantic equivalent of the official Python 3.6 implementation:
    # re.sub(' ', r'\s+', pattern). Python 3.7+ rejects \s in replacement
    # templates, so a callback preserves the same resulting regex string.
    return re.compile(re.sub(" ", lambda _: r"\s+", pattern), re.I | re.M)


def negbio_match_spans(text: str) -> list[Span]:
    """Use NegBio MIMIC-CXR RegExExtractor include/exclude semantics."""
    mentions, unmentions = _load_negbio_phrases()
    # The official extractor adds generated positional exclusions at startup.
    # Copy the cached dictionaries so per-report matching never mutates them.
    mentions = {name: list(values) for name, values in mentions.items()}
    unmentions = {name: list(values) for name, values in unmentions.items()}
    _add_positional_unmentions(mentions, unmentions, negbio=True)
    found: list[Span] = []
    for observation, phrases in mentions.items():
        excluded: list[tuple[int, int]] = []
        for phrase in unmentions.get(observation, []):
            excluded.extend(match.span() for match in _negbio_compile(phrase).finditer(text))
        for phrase in phrases:
            for match in _negbio_compile(phrase).finditer(text):
                if _overlaps_any(match.start(), match.end(), excluded):
                    continue
                found.append(Span(match.start(), match.end(), (f"negbio:{observation}",)))
    return found


@lru_cache(maxsize=1)
def _load_negbio_phrases() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    raw = yaml.safe_load(NEGBIO_PHRASES.read_text())
    mentions = {name: values.get("include", []) for name, values in raw.items()}
    unmentions = {name: values.get("exclude", []) for name, values in raw.items()}
    return mentions, unmentions


def expand_to_token(text: str, span: Span) -> Span:
    """Mask the full token after an official stem match (e.g. opaci -> opacities)."""
    start, end = span.start, span.end
    while start > 0 and WORD_CHAR_RE.fullmatch(text[start - 1]):
        start -= 1
    while end < len(text) and WORD_CHAR_RE.fullmatch(text[end]):
        end += 1
    return Span(start, end, span.sources)


def mask_label_mentions(text: str) -> tuple[str, list[Span]]:
    spans = chexpert_match_spans(text) + negbio_match_spans(text)
    spans = merge_spans(expand_to_token(text, span) for span in spans)
    cursor = 0
    pieces: list[str] = []
    for span in spans:
        pieces.append(text[cursor:span.start])
        pieces.append("___")
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces), spans


def transform_report(text: str, study_stem: str) -> tuple[str, bool, list[Span], list[Span]]:
    section_spans, lacks_sections = _section_removal_spans(text, study_stem)
    without_sections = remove_spans(text, section_spans)
    masked, mask_spans = mask_label_mentions(without_sections)
    return masked, lacks_sections, section_spans, mask_spans


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def canonical_reports(shard_root: Path) -> list[Path]:
    reports = [path for path in shard_root.glob("p*/s*.txt") if CANONICAL_REPORT_RE.fullmatch(path.name)]
    return sorted(reports, key=lambda path: (path.parent.name, path.name))


def write_missing_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["study_id", "subject_id", "report_path"])
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def run(args: argparse.Namespace) -> RunStats:
    if not SHARD_RE.fullmatch(args.shard):
        raise ValueError(f"Invalid shard {args.shard!r}; expected p10, p11, ..., p19")
    shard_root = args.reports_root.resolve() / args.shard
    if not shard_root.is_dir():
        raise FileNotFoundError(f"Shard directory does not exist: {shard_root}")

    reports = canonical_reports(shard_root)
    if args.limit is not None:
        reports = reports[: args.limit]
    stats = RunStats(shard=args.shard, discovered=len(reports), dry_run=args.dry_run)
    missing_rows: list[dict[str, str]] = []
    progress = ProgressReporter("mmp", args.shard, len(reports), getattr(args, "progress_every", 100))

    for report_number, report_path in enumerate(reports, start=1):
        match = CANONICAL_REPORT_RE.fullmatch(report_path.name)
        assert match is not None
        study_id = match.group("study_id")
        study_stem = f"s{study_id}"
        output_path = report_path.with_name(f"{study_stem}{REPORT_SUFFIX}.txt")
        try:
            original = report_path.read_text(encoding="utf-8")
            transformed, lacks_sections, section_spans, mask_spans = transform_report(original, study_stem)
            stats.processed += 1
            stats.section_characters_removed += sum(span.end - span.start for span in section_spans)
            stats.masked_spans += len(mask_spans)
            if lacks_sections:
                stats.without_findings_or_impression += 1
                missing_rows.append(
                    {
                        "study_id": study_id,
                        "subject_id": report_path.parent.name.removeprefix("p"),
                        "report_path": str(report_path),
                    }
                )
            if output_path.exists() and not args.overwrite and not args.dry_run:
                # Still parse every canonical report so an idempotent rerun
                # always rebuilds a complete audit CSV.
                stats.skipped_existing += 1
            elif not args.dry_run:
                atomic_write_text(output_path, transformed)
                stats.written += 1
        except Exception as exc:
            stats.errors += 1
            print(json.dumps({"error": str(exc), "report_path": str(report_path)}), flush=True)
            if args.fail_fast:
                raise
        finally:
            progress.update(report_number)

    audit_path = args.audit_csv or shard_root / AUDIT_CSV_NAME
    if not args.dry_run:
        write_missing_csv(audit_path, missing_rows)
    print(json.dumps({**asdict(stats), "audit_csv": str(audit_path)}, indent=2))
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard",
        dest="shards",
        required=True,
        nargs="+",
        help="One or more shards, e.g. --shard p10 p11 p12",
    )
    parser.add_argument(
        "--reports-root",
        type=Path,
        required=True,
        help=(
            "MIMIC-CXR free-text report root, i.e. the directory that contains "
            "the p10..p19 shard folders: <download>/physionet.org/files/mimic-cxr/2.1.0/files"
        ),
    )
    parser.add_argument("--audit-csv", type=Path, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=f"Replace existing *{REPORT_SUFFIX}.txt files",
    )
    parser.add_argument("--dry-run", action="store_true", help="Transform and count without writing")
    parser.add_argument("--limit", type=int, default=None, help="Testing aid: process at most N canonical reports")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after every N reports (default: 100)",
    )
    return parser


def run_many(args: argparse.Namespace) -> list[RunStats]:
    results: list[RunStats] = []
    for shard in args.shards:
        shard_args = copy(args)
        shard_args.shard = shard
        results.append(run(shard_args))
    return results


if __name__ == "__main__":
    run_many(build_parser().parse_args())
