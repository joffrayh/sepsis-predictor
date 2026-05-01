# Docstring Brief — `trajectory_builder.py`

**Module role**: Phase 3 of the pipeline. Consumes raw extracted CSVs and the Phase 2 cohort, and produces the final `sepsis_trajectories_4h.parquet` feature matrix. All functions are called from `build_trajectories`, which is the single public entry point.

---

## `load_measurement_mappings(path)`

**Summary**: Parses `clinical_reference/measurement_mappings.json` into three derived structures used downstream.

**Parameters**:
- `path` — filesystem path to `measurement_mappings.json`.

**Returns** (3-tuple):
- `measurements` — the raw dict keyed by concept name (e.g. `"heart_rate"`), each containing `"codes"` (list of MIMIC `itemid` ints) and optional `"hold_time"` (hours).
- `code_to_concept` — flat `dict[str, str]` mapping every individual `itemid` string → concept name. String keys because CSV `itemid` columns are read as strings.
- `hold_times` — `dict[str, int]` mapping concept name → maximum hours a measurement can be forward-filled by sample-and-hold. Only concepts with a `"hold_time"` entry appear here.

**Side effects**: None. Pure read.

**Invariants**: `code_to_concept` is built before `hold_times`; no ordering dependency between them.

---

## `load_and_filter_chunked(filepath, valid_stays, onset_df, time_col, winb4, winaft, itemid_filter, chunk_size)`

**Summary**: Streams a large pipe-delimited CSV in 1M-row chunks, dropping rows that belong to irrelevant stays or fall outside the time window of interest. Returns a single concatenated DataFrame. Returns an empty DataFrame (not an error) if the file is missing — callers decide how to handle that.

**Parameters**:
- `filepath` — absolute path to the CSV (pipe-delimited, `|`).
- `valid_stays` — set/array of `stay_id` values to keep. Applied first, before any other filter.
- `onset_df` — DataFrame with columns `stay_id`, `anchor_time` (Unix seconds). Pass `None` to skip time-window filtering.
- `time_col` — name of the timestamp column in the CSV to compare against `anchor_time`. Must be numeric (Unix seconds), not a datetime string.
- `winb4` — hours *before* `anchor_time` to include (default 24). Window start = `anchor_time − winb4×3600`.
- `winaft` — hours *after* `anchor_time` to include (default 72). Window end = `anchor_time + winaft×3600`. End is **exclusive** (`<`, not `<=`).
- `itemid_filter` — optional set of `itemid` strings to keep (used for `chartevents.csv` to drop unmapped codes early). `None` means no filter.
- `chunk_size` — rows per chunk (default 1,000,000). Tune down if RAM is tight.

**Returns**: Concatenated DataFrame of all retained rows, or empty DataFrame.

**Side effects**: Prints a tqdm progress bar per file.

**Invariants**: Time window filter is applied *only* when both `onset_df` and `time_col` are provided. The `anchor_time` column added during merge is dropped before appending to results — it does not appear in the output.

---

## `process_patient_measurements(ce_df, lab_df, mv_df, code_to_concept, batch_size, output_dir, chunk_size)`

**Summary**: Maps `itemid` codes to clinical concept names, concatenates chart events and lab events into a single long-format table, pivots to wide format (one row per `stay_id`+`charttime`, one column per concept), and merges mechanical ventilation status. Processes in `batch_size` stays at a time, flushing each batch to a temporary `.parquet` under `output_dir/_pivot_temp/` to cap RAM. Cleans up the temp directory before returning.

**Parameters**:
- `ce_df` — chartevents DataFrame with columns `stay_id`, `charttime`, `itemid`, `valuenum`. Must not be empty — raises `ValueError` if so.
- `lab_df` — labevents (union of `labs_ce` + `labs_le`, i.e. `labu`) with same columns. Must not be empty — raises `ValueError` if so.
- `mv_df` — mechanical ventilation DataFrame with columns `stay_id`, `charttime`, `mechvent`. Empty is acceptable (column is added as NaN).
- `code_to_concept` — `dict[str, str]` from `load_measurement_mappings`. Keys are `itemid` strings. Converted internally to `int` keys for efficient mapping.
- `batch_size` — number of stays per pivot batch (default 500).
- `output_dir` — directory under which `_pivot_temp/` is created. Must be writable.
- `chunk_size` — unused in this function (kept for API consistency with `load_and_filter_chunked`).

**Returns**: Wide DataFrame sorted by `(stay_id, charttime)` with columns: `stay_id`, `charttime`, one column per concept in `code_to_concept`, `mechvent`.

**Side effects**:
- Creates and then deletes `output_dir/_pivot_temp/pivot_consolidated.parquet`.
- Deletes `ce_df` and `lab_df` from memory once combined.
- Each batch's wide DataFrame is explicitly deleted after writing.

**Invariants**:
- The fixed column schema (`fixed_columns`) is derived from *all possible concepts* in `code_to_concept`, not just those present in the current data. This ensures every batch has identical columns so fastparquet can append cleanly.
- `pivot_table` uses `aggfunc="last"` — if two measurements for the same concept exist at the exact same `(stay_id, charttime)`, the last one (by row order) wins.
- `mv_df` empty is a warning, not an error — mechanical ventilation is sparse and may legitimately be absent.

---

## `standardise_patient_trajectories(init_traj, data_dict, onset, timestep, window_before, window_after, output_dir, flush_every)`

**Summary**: Projects every patient's irregular measurement timeline onto a fixed grid of `timestep`-hour bins, spanning `window_before` hours before to `window_after` hours after `anchor_time`. For each bin: clinical measurements are **averaged**; fluids and UO are **summed**; vasopressors use **median and max** of rates active in the window; antibiotic features are derived from overlap with the window. Results are flushed to disk every `flush_every` accumulated rows, then concatenated and returned. The temp directory is fully cleaned up.

**Parameters**:
- `init_traj` — wide DataFrame from `process_patient_measurements`. **Must be pre-sorted by `(stay_id, charttime)`** — the function uses `np.unique` with `return_index=True` to slice groups directly by position, so sort order is assumed correct.
- `data_dict` — `dict` with keys `"fluid"`, `"vaso"`, `"UO"`, `"abx"`, `"demog"`. Each value is a DataFrame. Keys are **popped** from the dict as they are consumed (the dict is mutated).
- `onset` — cohort DataFrame with columns `stay_id`, `anchor_time`, `onset_time`. Both time columns are Unix seconds (int).
- `timestep` — bin width in hours (default 4). Determines grid resolution.
- `window_before` — hours before `anchor_time` to include (default 24).
- `window_after` — hours after `anchor_time` to include (default 72). Total grid = `(window_before + window_after) / timestep` = 24 timesteps maximum (with default 4 h step).
- `output_dir` — writable directory for temp parquet chunks.
- `flush_every` — row count at which buffered rows are flushed to disk (default 5000).

**Returns**: DataFrame with one row per `(stay_id, timestep_idx)`. Columns: `timestep` (1-indexed bin number), `stay_id`, `onset_time`, `timestamp` (Unix seconds of bin start), demographics (`gender`, `age`, `charlson_comorbidity_index`, `re_admission`, `los`, `morta_hosp`, `morta_90`), all concept columns from `init_traj`, then fluid/vasopressor/UO/antibiotic aggregates.

**Side effects**:
- Pops `"fluid"`, `"vaso"`, `"UO"`, `"abx"`, `"demog"` from `data_dict` (irreversible mutation of the passed dict).
- Creates/deletes `output_dir/_std_temp/chunk_N.parquet` files.

**Invariants**:
- `RuntimeWarning: Mean of empty slice` is **intentionally suppressed** via `warnings.simplefilter("ignore", RuntimeWarning)` around `window_data.mean()`. This is expected when a timestep bin contains no measurements for a given column (all NaN) — it is not a data error.
- The `timestamp` column in the output is the bin *start* time (`window_start`), not the midpoint.
- `hours_since_first_abx` measures hours from the first antibiotic start to the *end* of the current window (`window_end`), not `window_start`.
- Stays with zero chart events are silently skipped (`len(ct_arr) == 0`).

---

## `build_trajectories(onset, valid_stays, data_dict, ce_df, lab_df, mv_df, code_to_concept, hold_times, config, output_dir)`

**Summary**: Orchestrates all Phase 3 sub-steps in the required order. This is the only public entry point for Phase 3.

**Parameters**:
- `onset` — cohort DataFrame (from Phase 2 `cohort.csv`) with `stay_id`, `anchor_time`, `onset_time`, `intime`.
- `valid_stays` — set of `stay_id` values in the cohort.
- `data_dict` — secondary tables dict (see `standardise_patient_trajectories`). Mutated during step 3.
- `ce_df`, `lab_df`, `mv_df` — DataFrames for chart events, lab events, mechvent (from `load_and_filter_chunked`).
- `code_to_concept` — from `load_measurement_mappings`.
- `hold_times` — from `load_measurement_mappings`.
- `config` — the `trajectories` section of `config.yaml` as a plain dict. All keys accessed with `[]` (no defaults — missing keys raise `KeyError`).
- `output_dir` — path to `processed_dir` for temp files and final output.

**Returns**: Final feature matrix DataFrame, or empty DataFrame if no valid trajectories.

**Sub-step execution order** (order is a hard invariant — do not reorder):

1. `process_patient_measurements` — pivot to wide format.
2. `handle_outliers` → `estimate_gcs_from_rass` → `estimate_fio2` → `handle_unit_conversions` → `sample_and_hold` — applied on the raw wide table *before* grid standardisation, so outlier and hold logic operates at the original measurement resolution.
3. `standardise_patient_trajectories` — project onto 4 h grid.
4. `add_missingness_features` — **must precede imputation** (step 5). If called after, all missingness indicators would be 0.
5. `handle_missing_values` (KNN) — fills residual NaNs after sample-and-hold.
6. `calculate_derived_variables` — SOFA sub-scores, `pf_ratio`, `shock_index`, SIRS. **Must follow imputation** (uses imputed values).
7. `apply_exclusion_criteria` — removes stays with physiologically impossible values or early death.
8. `add_infection_and_sepsis_flag` → `add_septic_shock_flag` — labels. Must follow SOFA scores (step 6).

**Side effects**: Deletes `ce_df`, `lab_df`, `mv_df` after pivoting.
