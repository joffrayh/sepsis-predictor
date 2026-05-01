---
description: "Document a data_processing pipeline file using its Pipeline Oracle brief. Auto-locates the matching brief and runs the Doc Scribe documentation workflow."
name: "Document Pipeline File"
argument-hint: "Path to the pipeline source file, e.g. src/data_processing/utils/labels.py"
agent: "Doc Scribe"
tools: [read, search, edit, todo]
---

Document the pipeline file: `$args`

## Brief Discovery

The brief for this file lives at `.github/briefs/`. Derive the brief filename by taking the stem of the target file and appending `_brief.md`.

Known mappings for reference:
| Source file | Brief |
|---|---|
| `cohort_builder.py` | `cohort_builder_brief.md` |
| `extraction/extractor.py` | `extractor_brief.md` |
| `trajectory_builder.py` | `trajectory_builder_brief.md` |
| `utils/imputation.py` | `imputation_brief.md` |
| `utils/clinical_heuristics.py` | `clinical_heuristics_brief.md` |
| `utils/labels.py` | `labels_brief.md` |

If no brief exists for the given file, pause and tell the user before proceeding.

## Instructions

Follow the Doc Scribe workflow exactly:

1. Read the matched brief in full from `.github/briefs/`.
2. Read the full source file at `$args`.
3. Use `search` tools to trace any cross-file dependencies needed to understand the logic.
4. Evaluate every function and class: keep, improve, remove, or add documentation.
5. Apply all changes to the source file.
6. Re-read the file top to bottom to verify consistency of tone and detail.

Adhere to all Doc Scribe style rules: NumPy docstrings, sparse human-like inline comments, no robotic filler, no logic changes.
