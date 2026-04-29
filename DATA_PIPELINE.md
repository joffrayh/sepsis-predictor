# MIMIC-IV Sepsis Prediction — Data Pipeline

This document describes how to set up the environment and run the full data processing pipeline, from downloading the raw MIMIC-IV dataset through to a final ML-ready Parquet file.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11.x | Exactly 3.11 (< 3.12) |
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
3. Ensure the extracted layout looks like this (the `.csv.gz` files must remain gzipped — the loader streams them directly):
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

This creates a `.venv/` and installs all dependencies declared in `pyproject.toml` (pandas, numpy, scikit-learn, lightgbm, mlflow, etc.).

---

## Step 2 — Run the Data Pipeline

The pipeline is a single Python entry point that handles all three phases end-to-end, with idempotent checkpointing so that completed phases are not repeated on re-runs.

```bash
cd /path/to/dissertation/code
source .venv/bin/activate

# First run — extract from raw MIMIC-IV files:
python src/data_processing/main_pipeline.py \
    --raw-data-dir data/raw/mimic-iv-3.1

# Subsequent runs — skip extraction (CSVs already exist):
python src/data_processing/main_pipeline.py
```

`--raw-data-dir` is only needed the **first time** (Phase 1). On subsequent runs it can be omitted and the pipeline picks up from the last completed checkpoint.

---

## Pipeline Phases

### Phase 1 — Extraction

*Skipped automatically if `--raw-data-dir` is not supplied.*

`MIMICExtractor` (in `processing_pipeline/extraction/extractor.py`) opens an in-process DuckDB connection, registers every `.csv.gz` file under `hosp/` and `icu/` as a view under the `mimiciv_hosp` / `mimiciv_icu` schemas, then runs 13 parameterised SQL queries defined in `processing_pipeline/extraction/extraction_metadata.json`. No PostgreSQL server is required — DuckDB reads directly from the gzipped source files.

Outputs to `data/processed_files/`:

| File | Contents |
|---|---|
| `cohort.csv` | ICU stays meeting basic inclusion criteria |
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

All files use `|` as the field separator.

---

### Phase 2 — Static Cohort Generation

*Checkpointed: skipped if `data/processed_files/cohort.csv` already exists.*

`cohort_builder.py` combines the raw extracted tables to produce a clean cohort definition:

1. **Microbiology fusion** — merges `microbio.csv` and `culture.csv`; fills missing `charttime` from `chartdate`.
2. **Demographics cleaning** — fills missing mortality flags and Charlson comorbidity index; deduplicates on `(admittime, dischtime)`.
3. **Readmission calculation** — marks stays where the patient was readmitted to ICU within 30 days.
4. **Antibiotic processing** — combines `abx.csv` (administered) and `bacterio_processed.csv`; deduplicates.
5. **Infection onset imputation** — for each stay, estimates a suspected infection onset time (`anchor_time`) from the first antibiotic and the corresponding positive culture result.
6. **Lab union** — concatenates `labs_ce.csv` and `labs_le.csv` into a single `labu.csv`.
7. **Outputs** — saves `cohort.csv`, `demog_processed.csv`, `abx_processed.csv`, and `labu.csv` to `data/processed_files/`.

`cohort.csv` has columns: `stay_id`, `subject_id`, `anchor_time`, `onset_time`, demographics.

---

### Phase 3 — Trajectory & Feature Engineering

This is the heaviest phase. It reads measurements in chunks to stay memory-efficient, then produces a single standardised timeseries per patient.

#### 3a. Chunked Loading & Time-window Filtering

`load_and_filter_chunked()` reads `chartevents.csv`, `labu.csv`, `mechvent.csv`, and all secondary tables (fluid, vaso, UO, abx, demog) in 1 million-row chunks. Each chunk is filtered to:
- Only valid `stay_id`s in the cohort.
- Only rows within a **24-hour window before** and **72-hour window after** the suspected infection onset (`anchor_time`).

#### 3b. Measurement Pivoting

`process_patient_measurements_vectorized()` maps all `itemid` codes to clinical concept names using `processing_pipeline/ReferenceFiles/measurement_mappings.json`, then pivots the long-format event table into a wide table with one row per `(stay_id, charttime)`. Pivoting is done in batches of 500 stays and flushed to temporary `.parquet` files to avoid out-of-memory errors.

Mechanical ventilation events are merged in at this stage.

#### 3c. 4-Hour Grid Standardisation

`standardize_patient_trajectories()` aligns every patient's measurements onto a fixed **4-hour timestep grid** relative to their `anchor_time` (spanning 24 hours before to 72 hours after onset = 25 timesteps). For each grid cell:

- Clinical measurements are averaged across all raw readings that fall in the window.
- Fluid, vasopressor, and urine output are summed over the window.
- Antibiotic flags (`abx_given`, `hours_since_first_abx`, `num_abx`) are computed.
- Static demographic features (age, gender, Charlson index, mortality flags, etc.) are attached.

Results are flushed to disk every 5,000 rows to keep memory bounded.

#### 3d. Outlier Handling & Cleaning

`handle_outliers()` applies bounds defined in `cleaning_config.json`. For each clinical variable, it can:

- Nullify values below `min_valid` or above `max_valid`.
- Clip values to `clip_low` or `clip_high`.
- Apply `log1p` transformation (used for `wbc`).

Three variables receive hardcoded special-case logic (not in the JSON config):

| Variable | Rule |
|---|---|
| `spo2` | Values > 150 → `NaN`; values in (100, 150] → clipped to 100 |
| `temp_C` | Values > 90 are assumed to be Fahrenheit; rescued into `temp_F` column, then nullified |
| `fio2` | Values > 100 → `NaN`; values < 1 multiplied by 100 (0–1 scale → percent); values < 20 → `NaN` |

`handle_unit_conversions()` then converts `temp_F` values to Celsius and merges them back into `temp_C`.

#### 3e. Missingness Features

Before imputation, `add_missingness_features()` adds two tracking columns for each of the key lab variables (`lactate`, `wbc`, `creatinine`, `platelets`):

- `<lab>_measured` — binary flag: was a real measurement present at this timestep?
- `hours_since_<lab>` — hours elapsed since the last valid measurement (NaN if never measured).

These features encode the *pattern* of measurement, which is clinically informative.

#### 3f. Imputation

1. **Sample-and-hold** (`sample_and_hold()`) — forward-fills each vital/lab value up to its clinical hold time (e.g., 8 hours for a blood gas, 24 hours for a full metabolic panel). This replicates the clinical assumption that the last recorded value is still valid until the next measurement.
2. **KNN imputation** (`handle_missing_values()`) — for values still missing after sample-and-hold, a KNN imputer fills remaining gaps using the closest neighbours across all features.

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
| `sofa_renal` | SOFA renal sub-score (0–4, from creatinine / UO) |
| `sofa_score` | Total SOFA score (sum of above) |
| `sirs_score` | SIRS criteria count (0–4, from temp, HR, RR/PaCO₂, WBC) |

#### 3h. Sepsis & Septic Shock Labels

- **`add_infection_and_sepsis_flag()`** — marks each timestep with `infection` (within the defined onset window) and `sepsis` (infection + SOFA ≥ 2).
- **`add_septic_shock_flag()`** — marks `septic_shock` where the Sepsis-3 criteria are met: sepsis + MAP < 65 mmHg + lactate > 2 mmol/L + rolling 12-hour fluid ≥ 2,000 mL (despite resuscitation), requiring vasopressors.

#### 3i. Exclusion Criteria

`apply_exclusion_criteria()` removes entire stays where:

1. `uo_step > 12,000 ml` in any 4-hour window (physiologically implausible).
2. `fluid_step > 10,000 ml` in any 4-hour window (physiologically implausible).
3. Hospital death within 24 hours of the first recorded measurement (too early to capture useful trajectories).

---

## Output

The final dataset is saved to:

```
data/processed_files/patient_timeseries_cleaned_final.parquet
```

Each row is one 4-hour timestep for one patient. Key columns:

| Column | Description |
|---|---|
| `stay_id` | ICU stay identifier |
| `timestep` | Index 1–25 (1 = 24h before onset, 13 = onset, 25 = 72h after) |
| `timestamp` | Unix timestamp of the grid cell start |
| `onset_time` | Suspected infection onset time |
| `gender`, `age` | Demographics |
| `charlson_comorbidity_index` | Comorbidity burden |
| `morta_hosp`, `morta_90` | Outcome labels |
| `<vital/lab>` | Cleaned, imputed clinical measurements |
| `<lab>_measured` | Missingness indicator (pre-imputation) |
| `hours_since_<lab>` | Time since last real measurement |
| `fluid_step`, `fluid_total` | Cumulative and windowed IV fluids (ml) |
| `uo_step`, `uo_total` | Cumulative and windowed urine output (ml) |
| `balance` | `fluid_total - uo_total` |
| `vaso_median`, `vaso_max` | Vasopressor dose statistics |
| `abx_given` | Binary: antibiotic given in this window |
| `sofa_score` | Total SOFA score |
| `sirs_score` | SIRS criteria count |
| `sepsis` | Binary sepsis label |
| `septic_shock` | Binary septic shock label |

---

## Re-running from a Checkpoint

The pipeline checks for the existence of intermediate files before each phase. To force a phase to re-run, delete its output file:

| Phase | Checkpoint file to delete |
|---|---|
| Phase 1 (Extraction) | Any file in `data/processed_files/` (e.g. `cohort.csv`) |
| Phase 2 (Cohort) | `data/processed_files/cohort.csv` |
| Phase 3 (Trajectories) | `data/processed_files/patient_timeseries_cleaned_final.parquet` |

---

## Linting

After any code changes to the pipeline:

```bash
uv run ruff check src/data_processing --fix
uv run ruff format src/data_processing
```
