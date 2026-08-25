"""Unit tests for the leakage-filtered MIMIC-CXR report extraction pipeline.

Run from the repository root:

    pytest -q tests/test_mimic_report_extraction.py

These tests use tiny synthetic reports; no MIMIC data is required.
"""

import argparse
import csv

from medmap.datasets.mimic_cxr.report_extraction import (
    AUDIT_CSV_NAME,
    REPORT_SUFFIX,
    chexpert_clean_with_mapping,
    mask_label_mentions,
    run,
    transform_report,
)


def test_removes_findings_and_impression_but_keeps_other_sections():
    report = (
        "FINAL REPORT\n INDICATION:  Fever and cough.\n \n"
        " FINDINGS:  Focal consolidation.\n \n"
        " IMPRESSION:  Pneumonia.\n"
    )
    output, lacks, section_spans, _ = transform_report(report, "s123")
    assert not lacks
    assert len(section_spans) == 1  # adjacent sections merge
    assert "Fever and cough" in output
    assert "FINDINGS" not in output
    assert "IMPRESSION" not in output
    assert "Pneumonia" not in output


def test_no_explicit_sections_is_audited_and_mentions_are_masked():
    report = "FINAL REPORT\n HISTORY:  Prior pneumothorax and opacities.\n"
    output, lacks, _, mask_spans = transform_report(report, "s123")
    assert lacks
    assert len(mask_spans) == 2
    assert "pneumothorax" not in output.lower()
    assert "opacities" not in output.lower()
    assert output.count("___") == 2


def test_unmention_exclusions_match_official_logic():
    output, spans = mask_label_mentions("There is a pericardial effusion.")
    assert "effusion" in output
    assert not any("Pleural Effusion" in source for span in spans for source in span.sources)


def test_partial_stem_masks_the_whole_token():
    output, _ = mask_label_mentions("Stable bibasilar opacities.")
    assert output == "Stable bibasilar ___."


def test_chexpert_clean_offsets_map_back_to_original():
    text = "No edema..  And/or opacity"
    cleaned, mapping = chexpert_clean_with_mapping(text)
    assert cleaned == "no edema. or opacity"
    start = cleaned.index("opacity")
    assert text[mapping[start][0] : mapping[start + len("opacity") - 1][1]] == "opacity"


def test_shard_write_is_idempotent_and_audit_remains_complete(tmp_path):
    subject = tmp_path / "p10/p10000001"
    subject.mkdir(parents=True)
    first = subject / "s1.txt"
    second = subject / "s2.txt"
    first_text = "REPORT\n FINDINGS:  Pneumonia.\n"
    first.write_text(first_text)
    second.write_text("REPORT\n HISTORY:  Prior pneumothorax.\n")
    args = argparse.Namespace(
        shard="p10",
        reports_root=tmp_path,
        audit_csv=None,
        overwrite=False,
        dry_run=False,
        limit=None,
        fail_fast=True,
    )

    initial = run(args)
    rerun = run(args)

    assert initial.written == 2
    assert rerun.skipped_existing == 2
    assert first.read_text() == first_text
    assert (subject / f"s1{REPORT_SUFFIX}.txt").read_text() == "REPORT\n"
    assert "pneumothorax" not in (subject / f"s2{REPORT_SUFFIX}.txt").read_text().lower()
    with (tmp_path / "p10" / AUDIT_CSV_NAME).open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["study_id"] for row in rows] == ["2"]
