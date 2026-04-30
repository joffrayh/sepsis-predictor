# Pipeline Configuration Files

This folder contains two JSON files that control the data cleaning and measurement mapping behaviour of the pipeline. Most users will not need to edit these, but they are the primary levers for adjusting clinical thresholds and adding new MIMIC-IV measurements.

---

## `cleaning_config.json`

Controls step 3d (outlier handling) in `utils/clinical_heuristics.py → handle_outliers()`.

Each key is a column name in the trajectory dataframe. The supported fields are:

| Field | Type | Effect |
|---|---|---|
| `min_valid` | number | Values **below** this → `NaN`. Use for physiologically impossible floors. |
| `max_valid` | number | Values **above** this → `NaN`. Use for physiologically impossible ceilings. |
| `clip_low` | number | Values **below** this → clipped to this value. Use for extreme-but-real outliers. |
| `clip_high` | number | Values **above** this → clipped to this value. Use for extreme-but-real outliers. |
| `transform` | string | Post-cleaning transformation applied after all bounds. Only `"log1p"` is supported (used for `wbc`). |

Rules are applied in this order: nullify (`min_valid`/`max_valid`) → clip (`clip_low`/`clip_high`) → transform.

### When to use `min_valid`/`max_valid` vs `clip_low`/`clip_high`

- **Nullify** (`min_valid`/`max_valid`) when the value is physically impossible and almost certainly a data entry error or unit mistake — e.g. a heart rate of 5 bpm.
- **Clip** (`clip_low`/`clip_high`) when the value is real but extreme enough to distort model training — e.g. a MAP of 220 mmHg is a genuine hypertensive crisis but would be an outlier; clipping to 180 preserves the signal that it was very high without letting a single value dominate.

### Example

```json
"heart_rate": { "min_valid": 7.5, "max_valid": 250, "clip_low": 20 }
```

| Raw value | Result |
|---|---|
| 5 bpm | → `NaN` (below `min_valid` 7.5) |
| 15 bpm | → clipped to 20 (below `clip_low`, but above `min_valid`) |
| 80 bpm | → 80 (unchanged) |
| 260 bpm | → `NaN` (above `max_valid` 250) |

### Hardcoded variables (not in this file)

Three variables have special-case logic in `handle_outliers()` that cannot be expressed as simple bounds and are therefore **not** config-driven:

| Variable | Rule |
|---|---|
| `spo2` | Values > 150 → `NaN`; values in (100, 150] → clipped to 100 |
| `temp_C` | Values > 90 assumed Fahrenheit; rescued into `temp_F` column, then nullified |
| `fio2` | Values > 100 → `NaN`; values < 1 multiplied by 100 (0–1 scale fix); values < 20 → `NaN` |

To change these rules, edit `utils/clinical_heuristics.py` directly.

---

## `measurement_mappings.json`

Drives two pipeline steps:

- **Step 3b** (`process_patient_measurements_vectorized`) — maps raw MIMIC-IV `itemid` codes to named concept columns during the pivot.
- **Step 3f** (`sample_and_hold`) — provides the `hold_time` per concept for forward-fill imputation.

Each key is the output column name. The supported fields are:

| Field | Type | Description |
|---|---|---|
| `codes` | list of strings | MIMIC-IV `itemid` values that map to this concept. Multiple codes mean the same measurement is recorded under different item IDs in different chart systems. |
| `display_name` | string | Human-readable label (informational only). |
| `unit` | string | Physical unit of the measurement (informational only). |
| `category` | string | Grouping: `vital`, `laboratory`, `blood_gas`, `respiratory`, `hemodynamic`, `neurological`, `demographic`. |
| `hold_time` | integer (hours) | Maximum duration to forward-fill this value during sample-and-hold imputation. |

### Hold times by category

Hold times reflect how long a measurement is considered clinically valid before a new reading is required:

| Category | Typical hold time | Rationale |
|---|---|---|
| `vital` | 2 h | Continuously monitored; stale after one timestep |
| `neurological` | 6–8 h | Assessed on shift rounds |
| `blood_gas` | 8 h | Ordered for acute changes; short validity |
| `laboratory` | 14–28 h | Routine panels ordered once or twice daily |
| `demographic` | 72–168 h | Height/weight essentially static during an ICU stay |

### Important: unlisted `itemid`s are silently dropped

Any raw chartevent or labevent row whose `itemid` does not appear in this file is discarded during the pivot step (step 3b). It will never appear as a column in the output trajectory. If you notice a clinical variable is missing from the final dataset, the first place to check is whether its MIMIC-IV `itemid` is present here.

### Adding a new measurement

1. Find the `itemid`(s) in MIMIC-IV (check `d_items` for chart events, `d_labitems` for lab events).
2. Add an entry to `measurement_mappings.json` with an appropriate `hold_time`.
3. If the new variable needs outlier bounds, add a corresponding entry to `cleaning_config.json`.
