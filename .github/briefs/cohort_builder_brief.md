# Docstring Brief — `cohort_builder.py`

**Module role**: Phase 2 of the pipeline. Reads raw extracted CSVs from `data/extracted/`, constructs the patient cohort with infection onset times, and writes processed intermediate files to `data/processed/`. Controlled entirely by `build_and_save_cohorts`, the single public entry point. Phase 2 is skipped entirely when `cohort.csv` already exists in `processed_dir` (checkpoint checked in `main.py`).

---

## `process_microbio_data(microbio, culture)`

**Summary**: Fuses the microbiology and culture DataFrames into a single bacteriology table. Fills missing `charttime` (exact event time) from `chartdate` (date-only fallback) for rows where only the date was recorded in MIMIC-IV. Drops `chartdate` after filling so only `charttime` remains.

**Parameters**:
- `microbio` — raw microbiology events DataFrame (from `microbio.csv`). Must have columns `charttime` and `chartdate`.
- `culture` — raw culture events DataFrame (from `culture.csv`).

**Returns**: Single concatenated DataFrame of all bacteriology events, with `charttime` filled where possible and `chartdate` dropped.

**Side effects**: Mutates `microbio["charttime"]` in place before concat.

**Invariants**: Rows where both `charttime` and `chartdate` are NaN remain NaN — no imputation is attempted.

---

## `process_demog_data(demog)`

**Summary**: Cleans the demographics DataFrame by filling missing outcome/comorbidity values with 0 and deduplicating on hospital admission window.

**Parameters**:
- `demog` — raw demographics DataFrame (from `demog.csv`).

**Returns**: Cleaned copy of the DataFrame, deduplicated on `(admittime, dischtime)`, keeping the first occurrence.

**Side effects**: None (returns a copy).

**Invariants**:
- `morta_90`, `morta_hosp`, `charlson_comorbidity_index` filled with 0 (not excluded) — missing outcomes are treated as no event/no comorbidity.
- Deduplication is on the hospital admission window `(admittime, dischtime)`, not on `stay_id` — a single hospital admission may span multiple ICU stays, but those are deduplicated at this step.

---

## `calculate_readmissions(demog, cutoff_days=30)`

**Summary**: Adds a binary `re_admission` flag to the demographics table. A patient is flagged as a readmission if the gap between their previous ICU discharge and current ICU admission is ≤ `cutoff_days` days.

**Parameters**:
- `demog` — cleaned demographics DataFrame (output of `process_demog_data`). Must have columns `subject_id`, `admittime`, `dischtime`. All times must be numeric (Unix seconds).
- `cutoff_days` — readmission window in days (default 30). Configurable via `config["readmission_window_days"]` in `config.yaml`.

**Returns**: Demographics DataFrame with `re_admission` column added (0/1) and the temporary `prev_dischtime` column dropped.

**Side effects**: Sorts `demog` by `(subject_id, admittime)` in place before shifting.

**Invariants**:
- Implementation is vectorised: `groupby("subject_id")["dischtime"].shift(1)` gives each row the previous discharge time within the same subject. No Python-level loop.
- First admission for each subject always gets `re_admission = 0` (`prev_dischtime` is NaN → mask is False).
- Comparison is in seconds: `cutoff = cutoff_days × 24 × 3600`.

---

## `fill_missing_icustay_ids(bacterio, demog, abx, stay_id_match_window_hours=48)`

**Summary**: Assigns `stay_id` to bacteriology and antibiotic events that were extracted without one. Matching logic: an event is assigned to an ICU stay if its timestamp falls within `stay_id_match_window_hours` hours of the stay's `intime`/`outtime`, **or** if the patient has only one ICU stay in the dataset (single-stay fallback).

**Parameters**:
- `bacterio` — bacteriology DataFrame (output of `process_microbio_data`). Rows with `stay_id = NaN` are the targets for filling. Uses `subject_id` as the join key.
- `demog` — cleaned demographics DataFrame with columns `subject_id`, `hadm_id`, `stay_id`, `intime`, `outtime`.
- `abx` — antibiotics DataFrame. Rows with any `stay_id` are candidates (the function uses `hadm_id` as the join key for ABx, not `subject_id`). Mutated in place.
- `stay_id_match_window_hours` — symmetric window around each stay's `intime`/`outtime` in which an event is considered to belong to that stay (default 48). Configurable via `config["stay_id_match_window_hours"]`.

**Returns**: 2-tuple `(bacterio, abx)` with `stay_id` filled where a match was found.

**Side effects**: Adds and drops a temporary `idx` column on both DataFrames.

**Invariants**:
- Bacteriology is matched on `subject_id`; ABx is matched on `hadm_id`. This reflects the MIMIC-IV schema where ABx events are keyed to hospital admissions.
- A known limitation (noted in code comments): the assignment overwrites `stay_id` using positional index alignment (`bacterio.loc[valid_bact["idx"], "stay_id"] = valid_bact["stay_id"].values`). If a missing `stay_id` row matches multiple stays, the last match wins for ABx (`.drop_duplicates(keep="last")`). This follows the logic of the original MIMIC-IV processing script.
- Rows that don't match any stay remain with `stay_id = NaN` and are filtered out in later steps.

---

## `find_infection_onset(abx, bacterio, abx_before_culture_hours=24, abx_after_culture_hours=72)`

**Summary**: Identifies the presumed infection onset time for each ICU stay using the Sepsis-3 operational definition. Onset is confirmed when an antibiotic administration and a culture sample are temporally proximate within the specified windows.

**Parameters**:
- `abx` — antibiotics DataFrame with columns `stay_id`, `starttime`. Rows with `stay_id = NaN` are excluded.
- `bacterio` — bacteriology DataFrame with columns `stay_id`, `subject_id`, `charttime`. Rows with `stay_id = NaN` are excluded.
- `abx_before_culture_hours` — max hours an antibiotic can precede a culture and still count as infection evidence (default 24). Sepsis-3 criterion 1.
- `abx_after_culture_hours` — max hours a culture can precede an antibiotic and still count (default 72). Sepsis-3 criterion 2.

**Returns**: DataFrame with columns `subject_id`, `stay_id`, `onset_time` — one row per stay, containing the earliest valid onset time.

**Side effects**: Prints the count of stays with a presumed onset.

**Onset time assignment rule** (Sepsis-3):
- If ABx was given **before** the culture (criterion 1): `onset_time = abx starttime`.
- If culture was taken **before** ABx (criterion 2): `onset_time = culture charttime`.
- Only the **earliest** valid onset per stay is kept (`sort_values("abx_idx").drop_duplicates(subset=["stay_id"], keep="first")`).

**Invariants**:
- `diff_hr` is an absolute time difference in hours; the direction (before/after) is determined separately by comparing `starttime` vs `charttime`.
- For each antibiotic event, only the *closest* culture is considered (`drop_duplicates(subset=["abx_idx"])` after sorting by `diff_hr`).
- Stays with no qualifying ABx+culture pair get no onset row — they appear in the final cohort with `onset_time = NaN`, meaning infection was not confirmed.

---

## `build_full_cohort(onset_df, demog)`

**Summary**: Merges presumed infection onset times onto the full ICU stay list. All stays (septic and non-septic) are included; non-septic stays receive `onset_time = NaN`. ICU `intime` is used as `anchor_time` for every stay, giving a consistent temporal reference point for Phase 3 grid alignment.

**Parameters**:
- `onset_df` — output of `find_infection_onset`. Contains `subject_id`, `stay_id`, `onset_time`.
- `demog` — processed demographics DataFrame with columns `subject_id`, `stay_id`, `intime`.

**Returns**: DataFrame with columns `subject_id`, `stay_id`, `anchor_time`, `intime`, `onset_time`. One row per unique ICU stay.

**Invariants**:
- `anchor_time = intime` for all stays — this is a deliberate design choice. Using ICU admission time as the anchor (rather than onset time) ensures Phase 3 time windows are aligned consistently whether or not infection was confirmed.
- `onset_time` is retained separately for downstream labelling (step 3h in Phase 3). It is *not* the anchor.
- Join is a left merge on `(subject_id, stay_id)` — all stays from `demog` are preserved.

---

## `build_and_save_cohorts(config, path_config)`

**Summary**: Orchestrates Phase 2 in full. Loads all required CSVs from `extracted_dir`, runs the processing pipeline in order, writes five output files to `processed_dir`, and returns the processed DataFrames.

**Parameters**:
- `config` — the `cohort` section of `config.yaml` as a plain dict. All keys accessed with `[]` (no defaults — missing keys raise `KeyError`). Required keys: `readmission_window_days`, `stay_id_match_window_hours`, `infection_abx_before_culture_hours`, `infection_abx_after_culture_hours`.
- `path_config` — the `paths` section of `config.yaml`. Required keys: `extracted_dir` (input), `processed_dir` (output).

**Returns**: 4-tuple `(cohort, bacterio, demog, data)`:
- `cohort` — final cohort DataFrame (written to `cohort.csv`).
- `bacterio` — processed bacteriology DataFrame (written to `bacterio_processed.csv`).
- `demog` — processed demographics DataFrame (written to `demog_processed.csv`).
- `data` — dict containing `"labU"` (lab union DataFrame, written to `labu.csv`) and `"abx"` (processed antibiotics, written to `abx_processed.csv`).

**Input files read** (all pipe-delimited `|`, from `extracted_dir`):
`abx.csv`, `culture.csv`, `microbio.csv`, `demog.csv`, `labs_ce.csv`, `labs_le.csv`

**Output files written** (all pipe-delimited `|`, to `processed_dir`):
`cohort.csv`, `bacterio_processed.csv`, `demog_processed.csv`, `labu.csv`, `abx_processed.csv`

**Sub-step execution order**:

1. `process_microbio_data` — fuse microbio + culture → bacteriology table.
2. `process_demog_data` — fill missing outcomes, deduplicate demographics.
3. `calculate_readmissions` — add `re_admission` flag.
4. `fill_missing_icustay_ids` — assign `stay_id` to unlinked bacteriology and ABx events.
5. `find_infection_onset` — identify Sepsis-3 infection onset per stay.
6. `build_full_cohort` — merge onset onto all stays, set `anchor_time = intime`.
7. Write all outputs, return.

**Notes on `labs_le` rename**: `labs_le.csv` uses `timestp` as the timestamp column name; it is renamed to `charttime` before concatenation with `labs_ce` so the union (`labu`) has a consistent column name throughout Phase 3.
