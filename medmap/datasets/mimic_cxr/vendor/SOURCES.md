# Vendored official source provenance

The pipeline uses pinned, unmodified parser and phrase resource files. Runtime
matching code in `extract_reports.py` follows the cited upstream extractor
implementations and records both sources on overlapping matches.

| Component | Revision | Vendored material | Official source |
|---|---|---|---|
| MIT-LCP/mimic-code | `5bdb9a0eb9f319c9b9dd6d533de33533ff3932e4` | `mimic-iv-cxr/txt/section_parser.py` | https://github.com/MIT-LCP/mimic-code/tree/5bdb9a0eb9f319c9b9dd6d533de33533ff3932e4/mimic-iv-cxr/txt |
| stanfordmlgroup/chexpert-labeler | `2f29daf2a7af97f494bc134e74600c05a7a5f987` | `phrases/mention`, `phrases/unmention` | https://github.com/stanfordmlgroup/chexpert-labeler/tree/2f29daf2a7af97f494bc134e74600c05a7a5f987 |
| ncbi-nlp/NegBio (`MIMIC-CXR`) | `962690b6789920fb0abab4fe05fc8ce6bc1a349d` | `patterns/chexpert_phrases.yml` | https://github.com/ncbi-nlp/NegBio/tree/962690b6789920fb0abab4fe05fc8ce6bc1a349d |

Relevant matching implementations:

- CheXpert `stages/extract.py` and `loader/load.py`
- NegBio `negbio/pipeline2/dner_regex.py`
- MIMIC `create_section_files.py`, including `custom_mimic_cxr_rules()`

The MIT-LCP execution scripts explicitly pass these CheXpert phrase directories
and the NegBio `chexpert_phrases.yml` file when generating MIMIC-CXR labels.
