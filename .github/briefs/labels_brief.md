# Docstring Brief — `utils/labels.py`

**Module role**: Step 3g–3h of Phase 3. Computes all derived clinical scores (SOFA sub-scores, SIRS, P/F ratio, shock index), assigns sepsis and septic shock labels using a 0/1/2 censoring scheme, and applies exclusion criteria to remove physiologically implausible stays. Called from `build_trajectories` after KNN imputation — all functions depend on imputed values being present.

---

## `calculate_derived_variables(df)`

**Summary**: Computes derived clinical features from imputed measurements. Must run **after** imputation (step 3f) because SOFA sub-scores and SIRS criteria use imputed values of `pf_ratio`, `platelets`, `bilirubin_total`, `map`, `gcs`, `creatinine`, `uo_step`, `temp_C`, `heart_rate`, `respiratory_rate`, `arterial_co2_pressure`, `wbc`.

**Parameters**:
- `df` — trajectory DataFrame after KNN imputation. Must contain all clinical measurement columns required for SOFA and SIRS scoring (see individual sub-score notes below).

**Returns**: Input DataFrame with the following columns added/modified: `gender`, `age`, `mechvent`, `charlson_comorbidity_index`, `vaso_median`, `vaso_max`, `pf_ratio`, `shock_index`, `sofa_resp`, `sofa_coag`, `sofa_liver`, `sofa_cv`, `sofa_cns`, `sofa_renal`, `sofa_score`, `sirs_score`.

**Side effects**: Mutates `df` in place (returns same object).

**Preprocessing applied before scoring** (not truly "derived variables" — these are corrections):
- `gender`: shifted by −1 (MIMIC encodes 1=F/2=M → becomes 0/1).
- `age > 150`: clamped to 91.4 (MIMIC-IV de-identification caps age at 91 for patients ≥ 91 years old; some rows encode this as a large number).
- `mechvent`: NaN → 0, then binarised (any positive value → 1).
- `charlson_comorbidity_index` NaN → column median.
- `vaso_median`, `vaso_max` NaN → 0.

**SOFA sub-scores** (each scored 0–4; NaN treated as 0 via `.fillna(0)`):
- `sofa_resp`: P/F ratio bins `[<100, <200, <300, <400, ≥400]` → `[4, 3, 2, 1, 0]`. Source: `arterial_o2_pressure` / (`fio2` / 100).
- `sofa_coag`: platelets (×10⁹/L) bins `[<20, <50, <100, <150, ≥150]` → `[4, 3, 2, 1, 0]`.
- `sofa_liver`: total bilirubin (mg/dL) bins `[<1.2, <2, <6, <12, ≥12]` → `[0, 1, 2, 3, 4]`.
- `sofa_cv`: MAP/vasopressor combination. Priority order via `np.select`: both NaN → 0; MAP ≥ 70 → 0; MAP 65–70 → 1; MAP < 65 → 2; `vaso_max ≤ 0.1` → 3; `vaso_max > 0.1` → 4.
- `sofa_cns`: GCS bins `[<6, <9, <12, <14, ≥14]` → `[4, 3, 2, 1, 0]`.
- `sofa_renal`: creatinine (mg/dL) takes priority over UO. Creatinine bins `[<1.2, <2, <3.5, <5, ≥5]` → `[0, 1, 2, 3, 4]`. UO fallback (mL/h): `uo_step` is a 4 h sum, so hourly rate = uo_step/4. Bins: ≥ 84 mL/h → 0; ≥ 34 → 3; < 34 → 4.

**`sofa_score`**: sum of all six sub-scores (0–24).

**SIRS score** (count of criteria met, 0–4):
- Temp: `temp_C ≥ 38` OR `temp_C ≤ 36`.
- HR: `heart_rate > 90`.
- RR: `respiratory_rate ≥ 20` OR `arterial_co2_pressure ≤ 32` (PaCO₂).
- WBC: `wbc ≥ 12` OR `wbc < 4` (×10⁹/L).

**`shock_index`**: `heart_rate / sbp_arterial`. Infinite values (zero SBP) → NaN, then column mean fills residual NaN.

---

## `apply_exclusion_criteria(df, exclusion_cfg)`

**Summary**: Removes entire ICU stays (all timesteps) that meet any of three exclusion criteria. Applied after derived variables and before labelling.

**Parameters**:
- `df` — trajectory DataFrame.
- `exclusion_cfg` — required dict (no default). Keys: `max_uo_per_window_ml`, `max_fluid_per_window_ml`, `early_death_hours`. All accessed with `[]`; missing keys raise `KeyError`. Populated from `config.yaml → trajectories → exclusion`.

**Returns**: Filtered DataFrame with excluded stays fully removed.

**Exclusion criteria** (all applied sequentially; each further filters the already-filtered result):
1. **Extreme UO**: any 4 h window with `uo_step > max_uo_per_window_ml` (default 12,000 mL). Physiologically impossible — flags bad sensor data.
2. **Extreme fluid**: any 4 h window with `fluid_step > max_fluid_per_window_ml` (default 10,000 mL). Same rationale.
3. **Early death**: hospital death (`morta_hosp = 1`) where the stay's entire recorded time span (last `timestamp` − first `timestamp`) is ≤ `early_death_hours` (default 24 h). These patients have insufficient trajectory data to be informative.

**Side effects**: Prints exclusion statistics (reason → count of stays excluded).

**Invariants**:
- Criteria are applied to distinct subsets of the original DataFrame sequentially. A stay excluded by criterion 1 will not be double-counted by criteria 2 or 3 (already absent).
- Early death calculation is based on `timestamp` span (Unix seconds / 3600), not on a mortality date column.

---

## `add_infection_and_sepsis_flag(df)`

**Summary**: Adds `infection_active` and `sepsis` columns using the Sepsis-3 definition. Uses a 0/1/2 censoring scheme for `sepsis` to distinguish onset from post-onset timesteps.

**Parameters**:
- `df` — trajectory DataFrame containing `onset_time`, `timestamp`, `stay_id`, `sofa_score`.

**Returns**: DataFrame with `infection_active` and `sepsis` columns added. Temporary `has_sepsis` column is dropped before return.

**Label definitions**:
- `infection_active` (0/1): 1 when `onset_time` is not NaN AND `timestamp ≥ onset_time`. Binary flag, no censoring.
- `sepsis` (0/1/2):
  - 0 = no sepsis (infection not active, or SOFA < 2, or pre-onset).
  - 1 = onset timestep: first timestep per stay where `infection_active = 1` AND `sofa_score ≥ 2`.
  - 2 = post-onset censored: all timesteps after the first sepsis onset (cumsum of `sepsis` == 1).

**Invariants**:
- Non-septic stays (`onset_time = NaN`) always have `sepsis = 0` for all timesteps — the `infection_active` check short-circuits to False.
- SOFA ≥ 2 alone is not sufficient; confirmed infection (known `onset_time`) is required.
- `has_sepsis` is a temporary helper column (cumulative sum used to propagate the censoring flag) and is dropped from the output.

---

## `add_septic_shock_flag(df, shock_cfg)`

**Summary**: Adds a `septic_shock` column using the Sepsis-3 definition of septic shock. Uses the same 0/1/2 censoring scheme as `sepsis`. Must be called **after** `add_infection_and_sepsis_flag` since it requires the `sepsis` column.

**Parameters**:
- `df` — trajectory DataFrame. Must contain `sepsis`, `map`, `lactate`, `vaso_max`, `fluid_step`, `stay_id`, `timestamp`.
- `shock_cfg` — required dict (no default). Keys: `map_threshold_mmhg`, `lactate_threshold_mmol`, `fluid_resuscitation_ml`, `fluid_window_timesteps`. Populated from `config.yaml → trajectories → septic_shock`.

**Returns**: DataFrame with `septic_shock` column added. Temporary `has_shock` and `fluid_rolling_12h` columns are dropped before return.

**Onset criteria** (all five must hold simultaneously at the same timestep):
1. `sepsis ∈ {1, 2}` — active sepsis already established.
2. `map < map_threshold_mmhg` (default 65 mmHg) — persistent hypotension.
3. `lactate > lactate_threshold_mmol` (default 2.0 mmol/L) — hyperlactataemia.
4. `vaso_max > 0` — vasopressors required.
5. Rolling `fluid_step` sum over `fluid_window_timesteps` consecutive 4 h windows ≥ `fluid_resuscitation_ml` (default: 3 × 4 h = 12 h window, ≥ 2,000 mL) — adequate volume resuscitation attempted.

**Label values** (same 0/1/2 scheme as `sepsis`):
- 0 = no shock.
- 1 = onset timestep (first timestep per stay meeting all five criteria).
- 2 = post-onset censored.

**Side effects**: Sorts `df` by `(stay_id, timestamp)` in place before computing rolling sum.

**Invariants**:
- `fluid_rolling_12h` is computed then dropped — it does not appear in the output.
- `has_shock` (cumsum helper) is also dropped from the output.
- A patient can have `septic_shock = 2` without ever having had `septic_shock = 1` in the output — this happens when the onset falls outside the observable trajectory window and only post-onset timesteps are present. This follows the same censoring design as `sepsis`.
