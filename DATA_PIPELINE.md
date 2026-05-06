# FRAMES Data Pipeline

This document describes how to set up the environment and run the full data processing pipeline, from downloading the raw MIMIC-IV dataset through to a final ML-ready Parquet file.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11.x | >=3.11,<3.12|
| [uv](https://docs.astral.sh/uv/) | latest | Package & venv manager |
| MIMIC-IV access | v3.1 | Requires PhysioNet credentialed access |

---

## Step 0 — Get MIMIC-IV Data

1. Request access to MIMIC-IV on [PhysioNet](https://physionet.org/content/mimiciv/). Credentialing is required.
2. Download the MIMIC-IV v3.1 zip archive (or use the PhysioNet CLI):
   ```bash
   wget -r -N -c -np --user <your_physionet_username> --ask-password \
       https://physionet.org/files/mimiciv/3.1/ \
       -P data/raw/
   ```
3. Ensure the extracted layout looks like this (the `.csv.gz` files must remain gzipped as the loader streams them directly):
   ```
   data/raw/mimic-iv-3.1/
   ├── hosp/
   │   ├── admissions.csv.gz
   │   ├── patients.csv.gz
   │   └── ...
   └── icu/
       ├── chartevents.csv.gz
       ├── icustays.csv.gz
       └── ...
   ```

---

## Step 1 — Install Python Dependencies

From the repository root:

```bash
uv sync
```
---

## Step 2 — Run the Data Pipeline

The pipeline has an orchestration script that handles all three phases end-to-end, with checkpointing so that completed phases are not repeated on re-runs.

```bash

# First run — extract from raw MIMIC-IV files:
uv run src/data_processing/main.py \
    --raw-data-dir data/raw/mimic-iv-3.1

# Subsequent runs — skip extraction (CSVs already exist):
uv run src/data_processing/main.py
```

`--raw-data-dir` is only needed the **first time** (Phase 1). On subsequent runs it can be left out and the pipeline picks up from the last completed checkpoint.

An optional `--config` flag specifies the pipeline configuration YAML (default: `src/data_processing/config.yaml`).

---

## Configuration

All pipeline behaviour is controlled by `src/data_processing/config.yaml`. Edit that file to change any default value. The file is divided into three sections:

| Section | Controls |
|---|---|
| `paths` | Input/output directories (`extracted_dir`, `processed_dir`), path to the extraction metadata JSON, and the final output filename |
| `cohort` | Readmission window (days), stay ID match window (hours), and the time windows for infection onset identification |
| `trajectories` | Time grid (`timestep`, `window_before`, `window_after`), imputation parameters (`missing_threshold`, `knn_neighbors`), exclusion thresholds, and performance tuning (`chunk_size`, `pivot_batch_size`, `flush_every_rows`) |

The `clinical_reference/` files (`measurement_mappings.json`, `outlier_bounds.json`) are likely not to be edited by most users. See [src/data_processing/clinical_reference/README.md](src/data_processing/clinical_reference/README.md) for details.

---

## Directory Layout

```
data/
├── raw/mimic-iv-3.1/        # Raw gzipped MIMIC-IV source files (input, read-only)
├── extracted/               # Phase 1 output CSVs
└── processed/               # Phase 2 & 3 output — intermediate CSVs + final Parquet

src/data_processing/
├── main.py                  # Pipeline orchestration script
├── config.yaml              # Configuration
├── cohort_builder.py        # Phase 2
├── trajectory_builder.py    # Phase 3
├── extraction/
│   ├── extractor.py         # Phase 1 DuckDB extractor
│   └── extraction_metadata.json  # 13 SQL queries
├── clinical_reference/
│   ├── measurement_mappings.json # itemid to concept name + sample-and-hold times
│   └── outlier_bounds.json       # Per-variable outlier bounds and transforms
└── utils/
    ├── clinical_heuristics.py    # Outlier handling, unit conversion, FiO2/GCS estimation
    ├── imputation.py             # Missingness features, sample-and-hold, KNN imputation
    └── labels.py                 # SOFA/SIRS scores, sepsis/shock labels, exclusion criteria
```

---

## Pipeline Phases

### Phase 1 — Extraction

*Skipped automatically if `--raw-data-dir` is not supplied.*

`MIMICExtractor` (in `extraction/extractor.py`) opens an in-process DuckDB connection, registers every `.csv.gz` file under `hosp/` and `icu/` as a view under the `mimiciv_hosp` / `mimiciv_icu` schemas, then runs 13 SQL queries defined in `extraction/extraction_metadata.json`.

Outputs to `data/extracted/`:

| File | Contents |
|---|---|
| `chartevents.csv` | Vital signs from ICU chart events |
| `labs_ce.csv` / `labs_le.csv` | Lab results (chart & lab event sources) |
| `demog.csv` | Patient demographics and comorbidities |
| `abx.csv` | Antibiotic administrations |
| `microbio.csv` / `culture.csv` | Microbiology culture results |
| `fluid.csv` | IV fluid administrations |
| `vaso.csv` | Vasopressor infusions |
| `uo.csv` | Urine output measurements |
| `mechvent.csv` | Mechanical ventilation events |
| `icustays.csv` | ICU stay metadata |
| `preadm_fluid.csv` | Pre-admission fluid intake |

If the output CSV already exists, that table is skipped.

---

### Phase 2 — Static Cohort Generation

*Checkpointed: skipped if `data/processed/cohort.csv` already exists.*

`cohort_builder.py` reads from `data/extracted/` and writes to `data/processed/`. Steps in order:

1. **Microbiology fusion**: merges `microbio.csv` and `culture.csv`; fills missing `charttime` from `chartdate`.
2. **Demographics cleaning**: fills missing mortality flags and Charlson comorbidity index with 0; deduplicates on `(admittime, dischtime)`.
3. **Readmission calculation**: flags stays where the patient was re-admitted to ICU within 30 days of their previous discharge.
4. **Missing `stay_id` filling**: matches bacteriology and ABx events to ICU stay windows (±48 h) to fill events that were extracted without a `stay_id`.
5. **Antibiotic processing**: merges `abx.csv` with the completed bacteriology table; deduplicates.
6. **Infection onset identification**: for each stay, identifies a presumed infection onset time (`onset_time`) using the Sepsis-3 definition: an antibiotic given ≤ 24 h before, or ≤ 72 h after, a positive culture. Only the earliest valid onset per stay is kept.
7. **Full cohort assembly**: merges onset times onto all ICU stays. ICU `intime` is used as `anchor_time` for every stay, regardless of whether infection was confirmed. Stays without a confirmed infection onset receive `onset_time = NaN`.
8. **Lab union**: concatenates `labs_ce.csv` and `labs_le.csv` into `labu.csv` (renames `timestp` to `charttime` in the LE table for consistency).

Outputs to `data/processed/`:

| File | Contents |
|---|---|
| `cohort.csv` | All ICU stays with `stay_id`, `subject_id`, `anchor_time` (= `intime`), `intime`, `onset_time` |
| `bacterio_processed.csv` | Fused bacteriology table with `stay_id` filled |
| `demog_processed.csv` | Cleaned demographics with `re_admission` flag |
| `abx_processed.csv` | Antibiotics with `stay_id` filled |
| `labu.csv` | Union of chart and lab event lab results |

---

### Phase 3 — Trajectory & Feature Engineering

This is the most intensive phase. Steps run in a fixed order. Sub-steps use `data/processed/` as a scratch directory for temporary Parquet files.

#### 3a. Chunked Loading & Time-window Filtering

`load_and_filter_chunked()` reads large CSVs in 1 million-row chunks. Each chunk is filtered to valid `stay_id`s and to rows within **24 h before** to **72 h after** `anchor_time` (ICU `intime`).

Sources and directories:
- From `data/extracted/`: `chartevents.csv`, `mechvent.csv`, `fluid.csv`, `vaso.csv`, `uo.csv`
- From `data/processed/`: `labu.csv`, `demog_processed.csv`, `abx_processed.csv`

#### 3b. Measurement Pivoting

`process_patient_measurements()` maps all `itemid` codes to clinical concept names using `clinical_reference/measurement_mappings.json`, concatenates chart events and lab events, then pivots the long-format table into a wide table with one row per `(stay_id, charttime)`. Pivoting is done in batches of 500 stays, flushed to temporary `.parquet` files to prevent OOM. The fixed column schema is derived from all possible concepts so every batch has identical columns. Mechanical ventilation events are merged in at this stage.

#### 3c. Outlier Handling, GCS/FiO₂ Estimation & Unit Conversion

Applied on the **raw wide table** before grid standardisation, so rules operate at original measurement resolution:

- `handle_outliers()` applies per-variable bounds from `clinical_reference/outlier_bounds.json`. Rules per variable (all optional): nullify below `min_valid` / above `max_valid`; clip to `clip_low` / `clip_high`; apply `log1p` transform.

  Three hardcoded special cases (not in the JSON):

  | Variable | Rule |
  |---|---|
  | `spo2` | > 150 → `NaN`; (100, 150] → clipped to 100 |
  | `temp_C` | > 90 assumed Fahrenheit — rescued into `temp_F` then nullified |
  | `fio2` | > 100 → `NaN`; < 1 → × 100 (0–1 scale fix); < 20 → `NaN` |

- `estimate_gcs_from_rass()` fills missing GCS from RASS using a fixed clinical mapping table.
- `estimate_fio2()` estimates missing FiO₂ from oxygen flow rate and device type using clinical approximation tables.
- `handle_unit_conversions()` cross-fills `temp_C`/`temp_F`, `hemoglobin`/`hematocrit`, and `bilirubin_total`/`bilirubin_direct` using empirical conversion formulas.
- `sample_and_hold()` forward-fills each vital/lab within each stay up to its maximum hold time (defined per concept in `measurement_mappings.json`, e.g. 8 h for blood gas, 24 h for metabolic panel).

#### 3d. 4-Hour Grid Standardisation

`standardise_patient_trajectories()` aligns every patient's measurements onto a fixed **4-hour timestep grid** spanning 24 h before to 72 h after `anchor_time` (ICU `intime`). For each grid cell:

- Clinical measurements are **averaged** across all raw readings in the window.
- Fluid, UO, and antibiotic counts are **summed**.
- Vasopressor dose: **median and max** of rates active in the window.
- Antibiotic flags (`abx_given`, `hours_since_first_abx`, `num_abx`) are computed from overlap with the window.
- Static demographic features are attached.

Results are flushed to disk every 5,000 accumulated rows to avoid memory issues.

#### 3e. Missingness Features

**Must run before imputation.** `add_missingness_features()` adds two columns per critical lab (`lactate`, `wbc`, `creatinine`, `platelets`):

- `<lab>_measured` — 1 if a real measurement was present at this timestep, 0 if NaN.
- `hours_since_<lab>` — hours since the last valid measurement within the same stay. Rows before the first measurement get the sentinel value `999.0`.

#### 3f. KNN Imputation

`handle_missing_values()` fills residual NaNs remaining after sample-and-hold (step 3c):

1. Columns with > 80% missingness are dropped entirely.
2. Columns with < 5% missingness: linear interpolation of internal gaps per patient.
3. Remaining columns: KNN imputation (`n_neighbors=1` by default) in patient-aligned chunks of ~10,000 rows to avoid imputing across patient boundaries.

#### 3g. Derived Variables & Scores

`calculate_derived_variables()` computes:

| Feature | Description |
|---|---|
| `pf_ratio` | Arterial O₂ pressure / (FiO₂ / 100) |
| `shock_index` | Heart rate / systolic BP |
| `sofa_resp` | SOFA respiratory sub-score (0–4, from P/F ratio) |
| `sofa_coag` | SOFA coagulation sub-score (0–4, from platelets) |
| `sofa_liver` | SOFA liver sub-score (0–4, from bilirubin) |
| `sofa_cv` | SOFA cardiovascular sub-score (0–4, from MAP + vasopressors) |
| `sofa_cns` | SOFA CNS sub-score (0–4, from GCS) |
| `sofa_renal` | SOFA renal sub-score (0–4, creatinine takes priority over UO) |
| `sofa_score` | Total SOFA score (0–24, sum of sub-scores) |
| `sirs_score` | SIRS criteria count (0–4, from temp, HR, RR/PaCO₂, WBC) |

#### 3h. Exclusion Criteria

`apply_exclusion_criteria()` removes entire stays where:

1. `uo_step > 12,000 mL` in any 4-hour window — physiologically implausible, flags bad sensor data.
2. `fluid_step > 10,000 mL` in any 4-hour window — same rationale.
3. Hospital death (`morta_hosp = 1`) within 24 hours of the first recorded measurement (insufficient trajectory length).

#### 3i. Sepsis & Septic Shock Labels

Both labels use a **0/1/2 censoring scheme**: 0 = criterion never met, 1 = onset timestep, 2 = post-onset.

- **`add_infection_and_sepsis_flag()`**
  - `infection_active` (0/1): 1 when `onset_time` is known AND `timestamp ≥ onset_time`.
  - `sepsis` (0/1/2): onset = first timestep where `infection_active = 1` AND `sofa_score ≥ 2`. Non-septic stays (no confirmed `onset_time`) always have `sepsis = 0`.

- **`add_septic_shock_flag()`**
  - `septic_shock` (0/1/2): onset = first timestep where **all five** of the following hold simultaneously:
    1. `sepsis ∈ {1, 2}` — active sepsis established.
    2. `map < 65 mmHg` — persistent hypotension.
    3. `lactate > 2.0 mmol/L` — hyperlactataemia.
    4. `vaso_max > 0` — vasopressors required.
    5. Rolling 12-hour fluid sum ≥ 2,000 mL (3 × 4-h windows) (adequate resuscitation attempted).

---

## Output

The final dataset is saved to:

```
data/processed/sepsis_trajectories_4h.parquet
```

Each row is one 4-hour timestep for one patient. Key columns:

| Column | Description |
|---|---|
| `stay_id` | ICU stay identifier |
| `timestep` | 1-indexed bin number (1 = first bin from 24 h before `anchor_time`) |
| `timestamp` | Unix timestamp of the grid cell start |
| `onset_time` | Presumed infection onset time (NaN for non-septic stays) |
| `gender`, `age` | Demographics |
| `charlson_comorbidity_index` | Comorbidity burden |
| `morta_hosp`, `morta_90` | Outcome labels |
| `<vital/lab>` | Cleaned, imputed clinical measurements |
| `<lab>_measured` | Missingness indicator (captured before imputation) |
| `hours_since_<lab>` | Hours since last real measurement (999.0 if never measured pre-timestep) |
| `fluid_step`, `fluid_total` | Windowed and cumulative IV fluids (mL) |
| `uo_step`, `uo_total` | Windowed and cumulative urine output (mL) |
| `balance` | `fluid_total − uo_total` |
| `vaso_median`, `vaso_max` | Vasopressor dose statistics (standardised rate) |
| `abx_given` | 1 if any antibiotic was active in this window |
| `hours_since_first_abx` | Hours from first antibiotic start to end of this window |
| `num_abx` | Count of distinct antibiotics active in this window |
| `sofa_score` | Total SOFA score |
| `sirs_score` | SIRS criteria count |
| `infection_active` | 1 if confirmed infection is active at this timestep |
| `sepsis` | 0 = no sepsis, 1 = onset timestep, 2 = post-onset |
| `septic_shock` | 0 = no shock, 1 = onset timestep, 2 = post-onset |

---

## Re-running from a Checkpoint

| Phase | How to force re-run |
|---|---|
| Phase 1 (Extraction) | Delete the relevant CSV(s) from `data/extracted/` |
| Phase 2 (Cohort) | Delete `data/processed/cohort.csv` |
| Phase 3 (Trajectories) | Delete `data/processed/sepsis_trajectories_4h.parquet` |

Phase 3 has no mid-phase checkpoint. A failure restarts from the beginning of Phase 3.

---

## Linting

After any code changes to the pipeline:

```bash
uv run ruff check src/data_processing --fix
uv run ruff format src/data_processing
```
