import json
import math
import os
import shutil
import warnings

import fastparquet
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from utils.clinical_heuristics import (
    estimate_fio2,
    estimate_gcs_from_rass,
    handle_outliers,
    handle_unit_conversions,
)
from utils.imputation import (
    add_missingness_features,
    handle_missing_values,
    sample_and_hold,
)
from utils.labels import (
    add_infection_and_sepsis_flag,
    add_septic_shock_flag,
    apply_exclusion_criteria,
    calculate_derived_variables,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_measurement_mappings(path):
    """
    Parse ``measurement_mappings.json`` into the three derived structures used by Phase 3.

    Parameters
    ----------
    path : str
        Path to ``clinical_reference/measurement_mappings.json``.

    Returns
    -------
    measurements : dict
        Raw mapping keyed by concept name (e.g. ``"heart_rate"``). Each entry
        contains ``"codes"`` (list of MIMIC ``itemid`` ints) and optionally
        ``"hold_time"`` (hours for sample-and-hold).
    code_to_concept : dict
        Flat ``{itemid_int: concept_name}`` lookup. Keys are Python ints, matching
        the int64 ``itemid`` dtype in the extracted CSVs.
    hold_times : dict
        ``{concept_name: max_hold_hours}`` for concepts that have a
        ``"hold_time"`` entry. Passed directly to ``sample_and_hold``.
    """
    with open(path) as f:
        measurements = json.load(f)

    code_to_concept = {}
    for concept, info in measurements.items():
        for code in info["codes"]:
            code_to_concept[code] = concept

    hold_times = {}
    for concept, info in measurements.items():
        if "hold_time" in info:
            hold_times[concept] = info["hold_time"]

    return measurements, code_to_concept, hold_times


def load_and_filter_chunked(
    filepath,
    valid_stays,
    onset_df=None,
    time_col=None,
    winb4=24,
    winaft=72,
    itemid_filter=None,
    chunk_size=1_000_000,
):
    """
    Stream a large pipe-delimited CSV in chunks, retaining only rows relevant to
    the target cohort and time window.

    Parameters
    ----------
    filepath : str
        Path to the pipe-delimited CSV to read.
    valid_stays : array-like
        ``stay_id`` values to keep. Applied to every chunk before any other filter.
    onset_df : pd.DataFrame, optional
        DataFrame with ``stay_id`` and ``anchor_time`` (Unix seconds). Required for
        time-window filtering; pass ``None`` to skip.
    time_col : str, optional
        Timestamp column in the CSV to compare against ``anchor_time``. Must be
        numeric Unix seconds. Required for time-window filtering.
    winb4 : int, optional
        Hours before ``anchor_time`` to include (default 24).
    winaft : int, optional
        Hours after ``anchor_time`` to include (default 72). Window end is
        exclusive (``<``, not ``<=``).
    itemid_filter : set of str, optional
        If provided, only rows whose ``itemid`` (cast to str) is in this set are
        kept. Used to drop unmapped codes from ``chartevents.csv`` early.
    chunk_size : int, optional
        Rows per read chunk (default 1,000,000).

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all retained rows, or an empty DataFrame if the
        file does not exist or no rows pass the filters.

    Notes
    -----
    Time-window filtering is only applied when both ``onset_df`` and ``time_col``
    are provided. The ``anchor_time`` column added during the merge is dropped
    before the chunk is appended — it does not appear in the output.
    """
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return pd.DataFrame()

    with open(filepath, "rb") as f:
        total_chunks = (sum(1 for _ in f) - 1) // chunk_size + 1

    chunks = []
    for chunk in tqdm(
        pd.read_csv(filepath, sep="|", chunksize=chunk_size, low_memory=False),
        total=total_chunks,
        desc=f"\tProcessing {os.path.basename(filepath)}",
        ncols=100,
    ):
        chunk = chunk[chunk["stay_id"].isin(valid_stays)]
        if chunk.empty:
            continue

        if itemid_filter is not None and "itemid" in chunk.columns:
            chunk = chunk[chunk["itemid"].astype(str).isin(itemid_filter)]
            if chunk.empty:
                continue

        if onset_df is not None and time_col is not None and time_col in chunk.columns:
            chunk = chunk.merge(
                onset_df[["stay_id", "anchor_time"]], on="stay_id", how="inner"
            )
            mask = (chunk[time_col] >= chunk["anchor_time"] - winb4 * 3600) & (
                chunk[time_col] < chunk["anchor_time"] + winaft * 3600
            )
            chunk = chunk[mask].drop(columns=["anchor_time"])

        chunks.append(chunk)

    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame()


def process_patient_measurements(
    ce_df,
    lab_df,
    mv_df,
    code_to_concept,
    batch_size=500,
    output_dir="data/processed_files",
    chunk_size=1_000_000,
):
    """
    Map ``itemid`` codes to concept names, pivot to wide format, and merge mechvent.

    Concatenates chart events and lab events into a long-format table, pivots to
    wide format (one row per ``stay_id`` + ``charttime``, one column per clinical
    concept), then merges mechanical ventilation status. Processes in ``batch_size``
    stays at a time, writing each batch to a temporary parquet file under
    ``output_dir/_pivot_temp/`` to bound peak RAM.

    Parameters
    ----------
    ce_df : pd.DataFrame
        Chartevents with columns ``stay_id``, ``charttime``, ``itemid``,
        ``valuenum``. Raises ``ValueError`` if empty.
    lab_df : pd.DataFrame
        Unified lab events (``labu``) with the same columns. Raises ``ValueError``
        if empty.
    mv_df : pd.DataFrame
        Mechanical ventilation with ``stay_id``, ``charttime``, ``mechvent``.
        An empty frame is accepted — ``mechvent`` will be all NaN.
    code_to_concept : dict
        ``{itemid_str: concept_name}`` from ``load_measurement_mappings``.
    batch_size : int, optional
        Stays per pivot batch (default 500).
    output_dir : str, optional
        Parent directory for the ``_pivot_temp/`` scratch folder.
    chunk_size : int, optional
        Unused; retained for API consistency with ``load_and_filter_chunked``.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame sorted by ``(stay_id, charttime)`` with one column per
        concept in ``code_to_concept`` plus ``mechvent``.

    Notes
    -----
    The fixed column schema for each batch is derived from *all possible concepts*
    in ``code_to_concept``, not just those present in the current batch. This ensures
    every batch has identical columns so ``fastparquet`` can append cleanly.

    ``pivot_table`` uses ``aggfunc="last"`` — when two measurements for the same
    concept share an exact ``(stay_id, charttime)``, the last one by row order wins.

    The long-format ``ce_lab`` frame is sorted on four columns *before* pivoting
    (cheap at long format) so the resulting wide frame is already ordered and
    ``sample_and_hold`` can skip its own expensive sort on the ~70-column wide table.
    """
    print("Pivoting chartevents, lab events and mechvent...")

    if ce_df.empty:
        raise ValueError(
            "chartevents dataframe is empty — check extracted_dir path and Phase 1 output."
        )
    print("\tMapping chartevents itemids to concepts...")
    int_code_to_concept = {int(k): v for k, v in code_to_concept.items()}
    ce_df["concept"] = ce_df["itemid"].map(int_code_to_concept)

    if lab_df.empty:
        raise ValueError(
            "labevents (labu) dataframe is empty — check extracted_dir path and Phase 1 output."
        )

    print("\tMapping labevents itemids to concepts...")
    lab_df["concept"] = lab_df["itemid"].map(int_code_to_concept)

    print("\tJoining chartevents and labevents...")
    ce_lab = pd.concat(
        [
            ce_df[["stay_id", "charttime", "concept", "valuenum"]]
            if not ce_df.empty
            else pd.DataFrame(),
            lab_df[["stay_id", "charttime", "concept", "valuenum"]]
            if not lab_df.empty
            else pd.DataFrame(),
        ]
    )
    ce_lab = ce_lab.dropna(subset=["concept"])

    del ce_df, lab_df

    # Sort on the 4-column long frame (cheap) so the pivoted wide frame comes out
    # ordered — sample_and_hold can then skip its expensive sort on the ~70-column table
    ce_lab.sort_values(["stay_id", "charttime"], inplace=True)
    ce_lab.reset_index(drop=True, inplace=True)

    all_stay_ids = ce_lab["stay_id"].unique()
    batches = [
        all_stay_ids[i : i + batch_size]
        for i in range(0, len(all_stay_ids), batch_size)
    ]
    print(f"\tPivoting in {len(batches)} batches of up to {batch_size} stays...")

    temp_dir = os.path.join(output_dir, "_pivot_temp")
    os.makedirs(temp_dir, exist_ok=True)
    consolidated_path = os.path.join(temp_dir, "pivot_consolidated.parquet")
    first_batch = True

    # derive fixed column schema from all possible concepts so every batch
    # has identical columns regardless of which concepts appear in that batch
    all_concepts = sorted(set(code_to_concept.values()))
    fixed_columns = ["stay_id", "charttime"] + all_concepts

    for i, batch_stays in enumerate(
        tqdm(batches, desc="\tPivoting batches", ncols=100)
    ):
        batch = ce_lab[ce_lab["stay_id"].isin(batch_stays)]
        if batch.empty:
            continue

        wide = batch.pivot_table(
            index=["stay_id", "charttime"],
            columns="concept",
            values="valuenum",
            aggfunc="last",
        ).reset_index()
        wide.columns.name = None

        wide = wide.reindex(columns=fixed_columns)

        fastparquet.write(
            consolidated_path,
            wide,
            compression="ZSTD",
            append=not first_batch,
        )
        first_batch = False
        del wide, batch

    del ce_lab

    print("\tReading consolidated pivot file from disk...")
    wide_data = pd.read_parquet(consolidated_path, engine="fastparquet")

    shutil.rmtree(temp_dir)

    # add mechvent
    if not mv_df.empty:
        print("\tMerging mechvent with wide data...")
        mv_clean = mv_df[["stay_id", "charttime", "mechvent"]].drop_duplicates(
            subset=["stay_id", "charttime"], keep="last"
        )
        del mv_df

        mv_map = mv_clean.set_index(["stay_id", "charttime"])["mechvent"]
        del mv_clean

        idx = pd.MultiIndex.from_arrays([wide_data["stay_id"], wide_data["charttime"]])
        wide_data["mechvent"] = idx.map(mv_map)
    else:
        print("WARNING: mechvent dataframe is empty.")
        wide_data["mechvent"] = np.nan

    return wide_data


def standardise_patient_trajectories(
    init_traj,
    data_dict,
    onset,
    bin_hours=4,
    window_before=24,
    window_after=72,
    output_dir="data/processed_files",
    flush_every=5000,
):
    """
    Project each patient's irregular measurement timeline onto a fixed-width time grid.

    Each ``timestep``-hour bin is populated with the mean of clinical measurements,
    the sum of fluids and UO, the median and max of active vasopressor rates, and
    antibiotic features derived from overlap with the window. Results are buffered
    and flushed to disk in ``flush_every``-row increments, then concatenated.

    Parameters
    ----------
    init_traj : pd.DataFrame
        Wide DataFrame from ``process_patient_measurements``. **Must be pre-sorted
        by ``(stay_id, charttime)``** — group boundaries are derived from
        ``np.unique`` with ``return_index=True``, which assumes contiguous groups.
    data_dict : dict
        Secondary tables with keys ``"fluid"``, ``"vaso"``, ``"UO"``, ``"abx"``,
        ``"demog"``. Each key is **popped** as it is consumed — the dict is mutated
        and partially emptied after this call returns.
    onset : pd.DataFrame
        Cohort with ``stay_id``, ``anchor_time``, and ``onset_time`` (Unix seconds).
    bin_hours : int, optional
        Bin width in hours (default 4).
    window_before : int, optional
        Hours before ``anchor_time`` to include (default 24).
    window_after : int, optional
        Hours after ``anchor_time`` to include (default 72).
    output_dir : str, optional
        Writable directory for ``_std_temp/chunk_N.parquet`` scratch files.
    flush_every : int, optional
        Row count at which the in-memory buffer is flushed to disk (default 5000).

    Returns
    -------
    pd.DataFrame
        One row per ``(stay_id, timestep_idx)``. Columns: ``timestep`` (1-indexed
        bin number), ``stay_id``, ``onset_time``, ``timestamp`` (Unix seconds of
        bin start), demographics, all concept columns from ``init_traj``, then
        fluid, vasopressor, UO, and antibiotic aggregates.

    Notes
    -----
    ``RuntimeWarning: Mean of empty slice`` is intentionally suppressed inside the
    ``window_data.mean()`` call. An all-NaN column for a given bin is expected and
    is not a data error.

    ``hours_since_first_abx`` measures hours from the first antibiotic start to the
    *end* of the current window (``window_end``), not the window start.

    Stays with no chart events in ``init_traj`` are silently skipped.
    """
    print("Standardising trajectories to fixed time step...")

    # create hash maps for the secondary tables — pop each entry from data_dict
    # immediately after converting so the raw DataFrames are freed before the loop
    fluid_grp = (
        {k: v for k, v in data_dict.pop("fluid").groupby("stay_id")}
        if "fluid" in data_dict
        else {}
    )
    vaso_grp = (
        {k: v for k, v in data_dict.pop("vaso").groupby("stay_id")}
        if "vaso" in data_dict
        else {}
    )
    uo_grp = (
        {k: v for k, v in data_dict.pop("UO").groupby("stay_id")}
        if "UO" in data_dict
        else {}
    )
    abx_grp = (
        {k: v for k, v in data_dict.pop("abx").groupby("stay_id")}
        if "abx" in data_dict
        else {}
    )
    demog_dict = (
        data_dict.pop("demog").set_index("stay_id").to_dict("index")
        if "demog" in data_dict
        else {}
    )
    anchor_dict = onset.set_index("stay_id")["anchor_time"].to_dict()
    onset_time_dict = onset.set_index("stay_id")["onset_time"].to_dict()

    columns = [
        "timestep",
        "stay_id",
        "onset_time",
        "timestamp",
        "gender",
        "age",
        "charlson_comorbidity_index",
        "re_admission",
        "los",
        "morta_hosp",
        "morta_90",
        *[
            col
            for col in init_traj.columns
            if col not in ["timestep", "stay_id", "charttime"]
        ],
        "fluid_total",
        "fluid_step",
        "uo_total",
        "uo_step",
        "balance",
        "vaso_median",
        "vaso_max",
        "abx_given",
        "hours_since_first_abx",
        "num_abx",
    ]

    processed_data = []
    flush_count = 0
    temp_dir = os.path.join(output_dir, "_std_temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_paths = []

    # use np.unique-based group boundaries instead of groupby — avoids building
    # the full 85K-entry index dict that causes OOM before the first iteration.
    # init_traj is pre-sorted by (stay_id, charttime) so groups are contiguous.
    stay_ids_arr = init_traj["stay_id"].to_numpy()
    unique_stay_ids, first_occurrence = np.unique(stay_ids_arr, return_index=True)
    group_boundaries = np.append(first_occurrence, len(init_traj))
    del stay_ids_arr

    for i, stay_id in enumerate(
        tqdm(unique_stay_ids, desc="\tStandardising per stay_id", ncols=100)
    ):
        patient_traj = init_traj.iloc[group_boundaries[i] : group_boundaries[i + 1]]
        start_time = anchor_dict.get(stay_id, 0)
        onset_time = onset_time_dict.get(stay_id, np.nan)
        demographics = demog_dict.get(stay_id, {})

        fluid_data = fluid_grp.get(stay_id, pd.DataFrame())
        vaso_data = vaso_grp.get(stay_id, pd.DataFrame())
        uo_data = uo_grp.get(stay_id, pd.DataFrame())
        abx_data = abx_grp.get(stay_id, pd.DataFrame())
        first_abx_time = abx_data["starttime"].min() if not abx_data.empty else np.nan

        # no sort needed — init_traj is pre-sorted by (stay_id, charttime)
        ct_arr = patient_traj["charttime"].to_numpy()
        if len(ct_arr) == 0:
            continue

        first_time = max(ct_arr[0], start_time - window_before * 3600)
        last_time = min(ct_arr[-1], start_time + window_after * 3600)

        num_timesteps = math.ceil((last_time - first_time) / (bin_hours * 3600))

        for timestep_idx in range(num_timesteps):
            window_start = first_time + (timestep_idx * bin_hours * 3600)
            window_end = window_start + (bin_hours * 3600)

            if window_end < first_time or window_start > last_time:
                continue

            # measurements: binary search on sorted charttime
            lo = np.searchsorted(ct_arr, window_start, side="left")
            hi = np.searchsorted(ct_arr, window_end, side="left")
            window_data = patient_traj.iloc[lo:hi]
            if len(window_data) == 0:
                measurements = {
                    col: np.nan
                    for col in patient_traj.columns
                    if col not in ["stay_id", "charttime"]
                }
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    measurements = window_data.mean(axis=0, skipna=True).to_dict()

            # fluids
            fluid_total, fluid_step = 0, 0
            if not fluid_data.empty:
                fluid_step = fluid_data[
                    (fluid_data["starttime"] < window_end)
                    & (fluid_data["endtime"] >= window_start)
                ]["amount"].sum()
                fluid_total = fluid_data[fluid_data["endtime"] < window_end][
                    "amount"
                ].sum()

            # vasopressors
            vaso_median, vaso_max = 0, 0
            if not vaso_data.empty:
                v_mask = (vaso_data["starttime"] <= window_end) & (
                    vaso_data["endtime"] >= window_start
                )
                w_vaso = vaso_data[v_mask]
                if len(w_vaso) > 0:
                    vaso_median = w_vaso["rate_std"].median()
                    vaso_max = w_vaso["rate_std"].max()

            # urine output
            uo_total, uo_step = 0, 0
            if not uo_data.empty:
                uo_step = uo_data[
                    (uo_data["charttime"] >= window_start)
                    & (uo_data["charttime"] < window_end)
                ]["value"].sum()
                uo_total = uo_data[uo_data["charttime"] < window_end]["value"].sum()

            # antibiotics
            abx_given, hrs_first_abx, num_abx = 0, None, 0
            if not abx_data.empty:
                abx_mask = (abx_data["starttime"] <= window_end) & (
                    abx_data["stoptime"] >= window_start
                )
                w_abx = abx_data[abx_mask]
                if len(w_abx) > 0:
                    abx_given = 1
                    num_abx = len(w_abx["drug"].unique())
                if pd.notna(first_abx_time):
                    hrs_first_abx = (window_end - first_abx_time) / 3600

            timestep_data = {
                "timestep": timestep_idx + 1,
                "stay_id": stay_id,
                "timestamp": window_start,
                "onset_time": onset_time,
                **demographics,
                **measurements,
                "fluid_total": fluid_total,
                "fluid_step": fluid_step,
                "uo_total": uo_total,
                "uo_step": uo_step,
                "balance": fluid_total - uo_total,
                "vaso_median": vaso_median,
                "vaso_max": vaso_max,
                "abx_given": abx_given,
                "hours_since_first_abx": hrs_first_abx,
                "num_abx": num_abx,
            }

            timestep_data.pop("charttime", None)
            processed_data.append(timestep_data)

        # Flush accumulated rows to disk to keep peak RAM bounded
        if len(processed_data) >= flush_every:
            path = os.path.join(temp_dir, f"chunk_{flush_count}.parquet")
            pd.DataFrame(processed_data, columns=columns).to_parquet(
                path, compression="zstd"
            )
            temp_paths.append(path)
            processed_data = []
            flush_count += 1

    # flush any remaining rows
    if processed_data:
        path = os.path.join(temp_dir, f"chunk_{flush_count}.parquet")
        pd.DataFrame(processed_data, columns=columns).to_parquet(
            path, compression="zstd"
        )
        temp_paths.append(path)

    print("\tConcatenating standardised chunks...")
    result = pd.concat(
        [
            pd.read_parquet(p)
            for p in tqdm(temp_paths, desc="\tReading chunks", ncols=100)
        ],
        ignore_index=True,
    )

    for p in temp_paths:
        os.remove(p)
    os.rmdir(temp_dir)

    return result


def build_trajectories(
    onset,
    valid_stays,
    data_dict,
    ce_df,
    lab_df,
    mv_df,
    code_to_concept,
    hold_times,
    config,
    output_dir,
):
    """
    Orchestrate all Phase 3 sub-steps to produce the final feature matrix.

    This is the single public entry point for Phase 3. Calls are delegated to
    ``process_patient_measurements``, clinical heuristic helpers, grid
    standardisation, imputation, and labelling in a fixed order.

    Parameters
    ----------
    onset : pd.DataFrame
        Cohort from Phase 2 (``cohort.csv``) with ``stay_id``, ``anchor_time``,
        ``onset_time``, and ``intime``.
    valid_stays : array-like
        Unique ``stay_id`` values in the cohort.
    data_dict : dict
        Secondary tables (``"fluid"``, ``"vaso"``, ``"UO"``, ``"abx"``,
        ``"demog"``). Mutated in step 3.
    ce_df : pd.DataFrame
        Chartevents (from ``load_and_filter_chunked``).
    lab_df : pd.DataFrame
        Unified lab events — ``labu`` (from ``load_and_filter_chunked``).
    mv_df : pd.DataFrame
        Mechanical ventilation events.
    code_to_concept : dict
        From ``load_measurement_mappings``.
    hold_times : dict
        From ``load_measurement_mappings``.
    config : dict
        The ``trajectories`` section of ``config.yaml``. All keys are accessed
        with ``[]`` — missing keys raise ``KeyError``.
    output_dir : str
        Path to ``processed_dir`` for temp files and final parquet output.

    Returns
    -------
    pd.DataFrame
        Final feature matrix, or an empty DataFrame if no valid trajectories exist.

    Notes
    -----
    Step ordering is a hard invariant — do not reorder:

    1. ``process_patient_measurements`` — pivot to wide format.
    2. ``handle_outliers`` → ``estimate_gcs_from_rass`` → ``estimate_fio2`` →
       ``handle_unit_conversions`` → ``sample_and_hold`` — applied at original
       measurement resolution, before grid standardisation.
    3. ``standardise_patient_trajectories`` — project onto fixed time grid.
    4. ``add_missingness_features`` — **must precede imputation**; after imputation
       all missingness indicators would be zero.
    5. ``handle_missing_values`` — KNN imputation of residual NaNs.
    6. ``calculate_derived_variables`` — SOFA sub-scores, pf_ratio, shock_index,
       SIRS. **Must follow imputation** (operates on imputed values).
    7. ``apply_exclusion_criteria`` — removes physiologically invalid stays.
    8. ``add_infection_and_sepsis_flag`` → ``add_septic_shock_flag`` — labels.
       Must follow step 6 (uses SOFA scores).
    """
    # 1. Pivot
    init_traj = process_patient_measurements(
        ce_df,
        lab_df,
        mv_df,
        code_to_concept,
        batch_size=config["pivot_batch_size"],
        output_dir=output_dir,
    )
    del ce_df, lab_df, mv_df

    if init_traj.empty:
        print("No valid trajectories found matching criteria.")
        return pd.DataFrame()

    # 2. Clinical Heuristics & Sample-Hold
    init_traj = handle_outliers(
        init_traj,
        config_path=config["cleaning_config_path"],
    )
    init_traj = estimate_gcs_from_rass(init_traj)
    init_traj = estimate_fio2(init_traj)
    init_traj = handle_unit_conversions(init_traj)
    init_traj = sample_and_hold(init_traj, hold_times)

    # 3. Standardise Time Grids
    init_traj = standardise_patient_trajectories(
        init_traj,
        data_dict,
        onset,
        bin_hours=config["timestep"],
        window_before=config["window_before"],
        window_after=config["window_after"],
        output_dir=output_dir,
        flush_every=config["flush_every_rows"],
    )

    # 4. Integrate Missingness Features BEFORE Imputation
    init_traj = add_missingness_features(init_traj, timestep_hours=config["timestep"])
    # 5. Imputation
    init_traj = handle_missing_values(
        init_traj,
        missing_threshold=config["missing_threshold"],
        knn_neighbors=config["knn_neighbors"],
    )

    # 6. Derived Variables
    init_traj = calculate_derived_variables(init_traj)

    # 7. Exclusion Criteria
    init_traj = apply_exclusion_criteria(init_traj, exclusion_cfg=config["exclusion"])

    # 8. Labels
    init_traj = add_infection_and_sepsis_flag(init_traj)
    init_traj = add_septic_shock_flag(init_traj, shock_cfg=config["septic_shock"])

    return init_traj
