# Docstring Brief — `utils/clinical_heuristics.py`

**Module role**: Steps 3b (outlier handling, unit conversion, GCS/FiO₂ estimation) of Phase 3. Applied on the **raw wide table** immediately after pivoting, before grid standardisation. This ordering is intentional: outlier rules and hold times operate at the original measurement resolution, not on 4 h averages. All functions are called from `build_trajectories` in this order: `handle_outliers` → `estimate_gcs_from_rass` → `estimate_fio2` → `handle_unit_conversions` → `sample_and_hold`.

---

## `handle_outliers(df, config_path)`

**Summary**: Applies per-variable clinical bounding rules to the measurement DataFrame, then applies three hardcoded special-case rules for SpO₂, temperature, and FiO₂ that are not representable in the JSON config. Config-driven rules are loaded from `outlier_bounds.json`; hardcoded rules are always applied if the relevant columns are present.

**Parameters**:
- `df` — wide DataFrame (output of `process_patient_measurements`). Values are raw, unprocessed measurement numbers.
- `config_path` — path to `clinical_reference/outlier_bounds.json`. If the file does not exist, config-driven rules are skipped (a warning is printed) but hardcoded rules still run.

**Returns**: Same DataFrame with out-of-range values nullified or clipped.

**Side effects**: Mutates `df` in place. Prints a tqdm progress bar over config variables.

**Config-driven rules** (applied per variable, in this order within each variable):
1. **`min_valid` / `max_valid`**: values outside the physiologically plausible range → NaN.
2. **`convert_f_to_c`**: `[lower, upper]` range — values within that range assumed to be Fahrenheit, converted to Celsius in-place (`C = (F − 32) / 1.8`).
3. **`impute_extreme_nans`**: boolean — if true, forward-fill then back-fill within each stay (for variables that should rarely be truly missing, e.g. when sensors momentarily drop to 0).
4. **`clip_low` / `clip_high`**: hard clipping to clinical domain bounds (values outside are clipped, not nullified).
5. **`transform: "log1p"`**: applies `np.log1p(col)` to the entire column (used for heavily right-skewed lab values).

**Hardcoded special-case rules** (applied after the config loop, always):
- **SpO₂** (`spo2`): `spo2 > 150` → NaN (impossible); `spo2 ∈ (100, 150]` → clipped to 100 (plausible saturation, erroneous scale).
- **Temperature** (`temp_C`): `temp_C > 90` is assumed to be a Fahrenheit value recorded in the Celsius column. If `temp_F` exists and is NaN at that row, the value is rescued into `temp_F` first. Then `temp_C > 90` → NaN.
- **FiO₂** (`fio2`): `fio2 > 100` → NaN (impossible percentage); `fio2 < 1` → × 100 (0–1 scale fix); `fio2 < 20` → NaN (below room air — physiologically implausible in ICU).

**Invariants**:
- Config-driven rules use `.get()` on per-variable JSON dicts — all fields are optional by design. A missing field means that rule is skipped for that variable.
- The hardcoded SpO₂ rule is a two-step sequence: first nullify > 150, *then* clip (100, 150] → 100. Order matters — swapping them would incorrectly clip values that should be NaN.
- The temp rescue (`temp_F` copy before nullifying `temp_C`) is conditional on `temp_F` being NaN at that row, to avoid overwriting a valid Fahrenheit reading.

---

## `estimate_gcs_from_rass(df)`

**Summary**: Imputes missing GCS (Glasgow Coma Scale) values from RASS (Richmond Agitation-Sedation Scale) scores using a fixed clinical mapping table. GCS is required for SOFA CNS sub-scoring; RASS is commonly documented in ICUs when formal GCS is not recorded.

**Parameters**:
- `df` — wide DataFrame. Must have `gcs` and `richmond_ras` columns (or `gcs` absent, in which case it is created as all-NaN first).

**Returns**: Same DataFrame with `gcs` filled where it was NaN and `richmond_ras` was present.

**Side effects**: Adds `gcs` column if absent. Mutates `df` in place.

**RASS → GCS mapping** (clinical approximation):

| RASS | GCS |
|------|-----|
| 0    | 15  |
| 1    | 15  |
| 2    | 15  |
| 3    | 15  |
| 4    | 15  |
| −1   | 14  |
| −2   | 12  |
| −3   | 11  |
| −4   | 6   |
| −5   | 3   |

**Invariants**:
- Only fills rows where `gcs` is NaN (`mask = df["gcs"].isna()`). Existing GCS values are never overwritten.
- RASS values not in the mapping table result in NaN (pandas `.map()` behaviour for unmapped keys).

---

## `estimate_fio2(df)`

**Summary**: Estimates FiO₂ (fraction of inspired oxygen, as a percentage 21–100) from oxygen flow rate and delivery device type when the direct FiO₂ measurement is missing. Uses clinically established flow-to-FiO₂ approximation tables for each device category. Only fills rows where `fio2` is NaN.

**Parameters**:
- `df` — wide DataFrame. Must have columns `fio2`, `oxygen_flow`, `oxygen_flow_cannula_rate`, `oxygen_flow_rate`, `oxygen_flow_device`.

**Returns**: Same DataFrame with `fio2` filled where it was NaN and sufficient device/flow information was available.

**Side effects**: Mutates `df` in place. `flow` is computed as a local Series and never added to `df`.

**Device categories and estimation logic**:

| `oxygen_flow_device` | Category | Estimation |
|---|---|---|
| `"0"`, `"2"` | Nasal cannula / simple | Flow-to-FiO₂ table (1–15 L/min → 24–70%); if flow unknown → 21% (room air). |
| `"3"`–`"6"`, `"8"`–`"12"` | Face mask variants | Flow-to-FiO₂ table (4–15 L/min → 36–75%). |
| `"7"` | High-flow oxygen | Flow-to-FiO₂ table (≤6–≥15 L/min → 60–100%). |
| `"13"` | High-flow face mask | Flow-to-FiO₂ table (<10–≥15 L/min → 60–100%). |
| `"14"` | Simple face mask | Flow-to-FiO₂ table (<5–≥10 L/min → 40–80%). |

**Flow resolution priority**: `oxygen_flow` → `oxygen_flow_cannula_rate` → `oxygen_flow_rate` (first non-NaN wins, via `.fillna()` chaining).

**Invariants**:
- Tables are applied high-to-low: the loop iterates from the highest threshold downward, and each iteration overwrites `fio2` for rows `≤ threshold`. This means the last matching condition (smallest threshold ≥ actual flow) wins — equivalent to a range lookup.
- Only rows where `fio2` is currently NaN are eligible for estimation (each `mask` begins with `df["fio2"].isna()`).
- The function only fills by estimation — it does not nullify or clip values. Outlier nullification for FiO₂ happens earlier in `handle_outliers`.

---

## `handle_unit_conversions(df)`

**Summary**: Performs cross-column unit harmonisation for temperature, haemoglobin/haematocrit, and bilirubin. Uses empirical conversion formulas to fill missing values when the complementary measurement is available. Applied after `estimate_fio2` and before `sample_and_hold`.

**Parameters**:
- `df` — wide DataFrame. Must have columns `temp_C`, `temp_F`, `hemoglobin`, `hematocrit`, `bilirubin_total`, `bilirubin_direct`.

**Returns**: Same DataFrame with cross-filled missing values.

**Side effects**: Mutates `df` in place.

**Conversion rules applied** (in order):

1. **Celsius values in the Fahrenheit column**: `temp_F ∈ (25, 45)` → these are clearly Celsius values recorded in the wrong column. Move to `temp_C`, nullify `temp_F`.
2. **Fahrenheit values in the Celsius column**: `temp_C > 70` → these are clearly Fahrenheit. Move to `temp_F`, nullify `temp_C`.
3. **Fill `temp_F` from `temp_C`**: where `temp_C` is known and `temp_F` is NaN. Formula: `F = C × 1.8 + 32`.
4. **Fill `temp_C` from `temp_F`**: where `temp_F` is known and `temp_C` is NaN. Formula: `C = (F − 32) / 1.8`.
5. **Fill `hematocrit` from `hemoglobin`**: `Hct = (Hgb × 2.862) + 1.216`.
6. **Fill `hemoglobin` from `hematocrit`**: `Hgb = (Hct − 1.216) / 2.862`.
7. **Fill `bilirubin_direct` from `bilirubin_total`**: `Direct = (Total × 0.6934) − 0.1752`.
8. **Fill `bilirubin_total` from `bilirubin_direct`**: `Total = (Direct + 0.1752) / 0.6934`.

**Invariants**:
- Each rule is applied only when one side is NaN and the other is not — no values are ever overwritten.
- Rules 1–2 (rescue of cross-column mis-recordings) run before rules 3–4 (forward filling), so rescued values can themselves be used to fill missing entries in the same pass.
- The haemoglobin and bilirubin conversion formulas are empirical linear approximations, not exact physiological identities.
