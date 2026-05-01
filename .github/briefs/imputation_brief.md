# Docstring Brief — `utils/imputation.py`

**Module role**: Steps 3e–3f of Phase 3. Implements three distinct imputation/missingness strategies applied in strict order: (1) missingness feature engineering before any imputation, (2) sample-and-hold forward fill on the raw wide table, (3) KNN imputation on the standardised grid. Called from `build_trajectories`.

---

## `sample_and_hold(df, vitalslab_hold)`

**Summary**: Forward-fills clinical measurements within each ICU stay up to a per-concept maximum hold time. Simulates the clinical reality that a measurement remains valid until a new one is taken or until it is too stale to trust. Applied on the **raw wide table** (before grid standardisation) so hold logic operates at the original measurement resolution, not the 4 h grid.

**Parameters**:
- `df` — wide DataFrame sorted by `(stay_id, charttime)`. **Must be pre-sorted** (guaranteed by `process_patient_measurements`). Columns are concept names; rows are individual measurement events.
- `vitalslab_hold` — `dict[str, int]` mapping concept name → maximum hold time in hours. Only concepts present in both `vitalslab_hold` and `df.columns` and with numeric dtype are processed. Source: `hold_times` from `load_measurement_mappings`.

**Returns**: Same DataFrame with forward-filled values where the gap falls within the hold period.

**Side effects**: Modifies `df` in place (returns same object). Prints a tqdm progress bar per column.

**Invariants**:
- The fill logic uses two `groupby(...).transform("ffill")` passes: one to get the timestamp of the last valid measurement, one to get its value. The hold is applied only when: (a) the current value is NaN, AND (b) the time since the last valid measurement ≤ `hold_period_secs`.
- No fill is applied beyond the hold period — gaps larger than `hold_time` remain NaN for KNN to handle later.
- The sort order of `df` is trusted; no re-sorting occurs inside this function.

---

## `fixgaps(x)`

**Summary**: Linearly interpolates internal NaN gaps in a 1D NumPy array. Only fills NaNs that are **between** two valid (non-NaN) values. Leading and trailing NaN runs are left as NaN.

**Parameters**:
- `x` — 1D NumPy array (numeric). May contain NaN.

**Returns**: Copy of `x` with internal NaN gaps linearly interpolated. Leading/trailing NaNs unchanged.

**Side effects**: None. Pure function, returns a new array.

**Invariants**:
- If `x` contains zero valid values, returns an unchanged copy.
- Uses `scipy.interpolate.interp1d` on the indices of valid values to estimate interior NaN positions.
- Called as a `transform` lambda inside `handle_missing_values` — it operates on per-patient column slices.

---

## `handle_missing_values(df, missing_threshold=0.8, knn_neighbors=1)`

**Summary**: Three-stage residual imputation on the standardised 4 h grid, after sample-and-hold. Handles measurement columns only; metadata, treatment, and label columns are excluded from all imputation. Stages run in order: drop high-missingness columns → linear interpolation for low-missingness → KNN for moderate-missingness.

**Parameters**:
- `df` — trajectory DataFrame (output of `standardise_patient_trajectories`) sorted by `(stay_id, timestamp)`.
- `missing_threshold` — fraction above which a column is dropped entirely (default 0.8 = 80%). Configurable via `config["missing_threshold"]`.
- `knn_neighbors` — number of neighbours for KNN imputation (default 1). Configurable via `config["knn_neighbors"]`.

**Returns**: DataFrame with imputed measurement columns, high-missingness columns removed, and original column order otherwise preserved.

**Non-imputed columns** (excluded from all stages): `timestep`, `stay_id`, `onset_time`, `timestamp`, `gender`, `age`, `charlson_comorbidity_index`, `re_admission`, `los`, `morta_hosp`, `morta_90`, `fluid_total`, `fluid_step`, `uo_total`, `uo_step`, `balance`, `vaso_median`, `vaso_max`, `abx_given`, `hours_since_first_abx`, `num_abx`.

**Stage 1 — Column dropping**: Any measurement column with missingness ≥ `missing_threshold` is removed from the DataFrame entirely. Missingness is computed globally (not per-patient).

**Stage 2 — Linear interpolation** (per-patient, internal gaps only): Applied to columns with 0% < missingness < 5%. Uses `fixgaps` via `groupby("stay_id").transform(...)`. Only fills internal gaps within a patient's trajectory.

**Stage 3 — KNN imputation** (patient-aligned chunks): Applied to all remaining measurement columns with missingness ≥ 5% (after stage 2 has already handled the low-missingness ones). Builds patient-aligned chunks of ~9,999 rows (whole patients only — never splits a patient across chunks) so KNN never imputes across patient boundaries. A fresh `KNNImputer` is fit on each chunk independently.

**Side effects**: Re-sorts `df` by `(stay_id, timestamp)` at the start. Prints progress information.

**Invariants**:
- `keep_empty_features=True` is passed to `KNNImputer` so columns that are all-NaN within a chunk do not cause an error.
- Missingness fractions are computed on the pre-drop DataFrame — a column at exactly 80% is dropped (the threshold comparison is `< missing_threshold`, not `<=`).
- Patient-chunk boundaries are determined by `groupby("stay_id", sort=False)` — order is insertion order, not `stay_id` sort order. This is intentional to avoid unnecessary re-sorting.

---

## `add_missingness_features(df, lab_cols, timestep_hours=4)`

**Summary**: Adds binary "was this lab measured?" and "hours since last measurement" features for a set of critical labs. **Must be called before any imputation** (steps 3f onward). After imputation, NaN patterns are destroyed and these features would be meaningless.

**Parameters**:
- `df` — trajectory DataFrame (output of `standardise_patient_trajectories`). Must have `stay_id`, `timestep` columns, and the columns named in `lab_cols`.
- `lab_cols` — list of column names to generate missingness features for. Default: `["lactate", "wbc", "creatinine", "platelets"]`. These are the four labs most clinically relevant for SOFA scoring and sepsis detection.
- `timestep_hours` — duration of each timestep in hours (default 4). Used to convert timestep index differences to hours. Must match the grid resolution used in `standardise_patient_trajectories`.

**Returns**: DataFrame with two new columns per lab in `lab_cols` (if the column exists):
- `<lab>_measured` — float 0/1: 1 if the lab had a real value at this timestep (not NaN), 0 if imputed/missing.
- `hours_since_<lab>` — float: hours elapsed since the last real measurement within the same patient. 999.0 for all timesteps before the first measurement of that lab.

**Side effects**: Sorts `df` by `(stay_id, timestep)` in place at the start.

**Invariants**:
- The `hours_since_<lab>` calculation works by forward-filling the *timestep index* of the last valid measurement within each patient (`groupby("stay_id")[last_measured_step.name].ffill()`), then computing `(current_timestep − last_measured_step) × timestep_hours`. This gives elapsed hours in grid-time, not wall-clock time.
- Pre-first-measurement rows get 999.0 (sentinel value meaning "never measured"). This is not NaN — it is a deliberate large value that downstream models can treat as "very long ago".
- Labs not present in `df.columns` are silently skipped (no error).
- This function is **order-critical**: must run after `standardise_patient_trajectories` (to have the `timestep` column) and before `handle_missing_values` (to capture real missingness before KNN fills it).
