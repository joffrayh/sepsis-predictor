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
    Reads large CSV files in chunks and filters out rows not belonging
    to our target patients or target time windows.
    This saves huge RAM compared to old function.
    """
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return pd.DataFrame()

    with open(filepath, "rb") as f:
        total_chunks = (sum(1 for _ in f) - 1) // chunk_size + 1

    chunks = []
    # read in row chunks to keep memory usage small
    for chunk in tqdm(
        pd.read_csv(filepath, sep="|", chunksize=chunk_size, low_memory=False),
        total=total_chunks,
        desc=f"\tProcessing {os.path.basename(filepath)}",
        ncols=100,
    ):
        # get valid patients
        chunk = chunk[chunk["stay_id"].isin(valid_stays)]
        if chunk.empty:
            continue

        # filter by item IDs if provided (e.g. for chartevents)
        if itemid_filter is not None and "itemid" in chunk.columns:
            chunk = chunk[chunk["itemid"].astype(str).isin(itemid_filter)]
            if chunk.empty:
                continue

        # filter by time window if time column is provided
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
    print("Pivoting chartevents, lab events and mechvent...")

    # build int-keyed map once to avoid allocating object-dtype string Series per .map() call
    int_code_to_concept = {int(k): v for k, v in code_to_concept.items()}

    if ce_df.empty:
        raise ValueError(
            "chartevents dataframe is empty — check extracted_dir path and Phase 1 output."
        )
    print("\tMapping chartevents itemids to concepts...")
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

    # free the input frames now they're combined
    del ce_df, lab_df

    # sort on the 4-column long frame now (cheap) so wide_data comes out ordered
    # and sample_and_hold can skip its expensive sort on the 70-column wide frame
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
    all_concepts = sorted(set(int_code_to_concept.values()))
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

        # reindex to fixed schema so all batches have identical columns for append
        wide = wide.reindex(columns=fixed_columns)

        # append each batch into a single file to avoid loading all into RAM at once
        fastparquet.write(
            consolidated_path,
            wide,
            compression="ZSTD",
            append=not first_batch,
        )
        first_batch = False

        # explicitly free batch memory before next iteration
        del wide, batch

    del ce_lab

    print("\tReading consolidated pivot file from disk...")
    wide_data = pd.read_parquet(consolidated_path, engine="fastparquet")

    # clean up temp dir
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
    timestep=4,
    window_before=24,
    window_after=72,
    output_dir="data/processed_files",
    flush_every=5000,
):
    """
    Standardise patient trajectories to fixed time steps (e.g. every 4 hours)
    relative to ICU admission time (anchor_time).
    For each time step, we take the mean of all measurements within that time step.
    Processed data is flushed to disk every flush_every rows to avoid OOM errors.
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

    # for each patient trajectory, create fixed time steps relative to
    # ICU admission time (anchor_time) and estimate measurements within each step
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

        num_timesteps = math.ceil((last_time - first_time) / (timestep * 3600))

        for timestep_idx in range(num_timesteps):
            window_start = first_time + (timestep_idx * timestep * 3600)
            window_end = window_start + (timestep * 3600)

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

        # flush to disk once we've accumulated enough rows
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
    End-to-end timeseries sequence generation.
    Memory-safely pivots data, standardises time grids, imputes missing values and generates labels.
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
        timestep=config["timestep"],
        window_before=config["window_before"],
        window_after=config["window_after"],
        output_dir=output_dir,
        flush_every=config["flush_every_rows"],
    )

    # 4. Integrate Missingness Features BEFORE Imputation
    init_traj = add_missingness_features(
        init_traj, timestep_hours=config["timestep"]
    )
    # 5. Imputation
    init_traj = handle_missing_values(
        init_traj,
        missing_threshold=config["missing_threshold"],
        knn_neighbors=config["knn_neighbors"],
    )

    # 6. Labels & Derived Variables
    init_traj = calculate_derived_variables(init_traj)
    init_traj = apply_exclusion_criteria(
        init_traj, exclusion_cfg=config["exclusion"]
    )
    init_traj = add_infection_and_sepsis_flag(init_traj)
    init_traj = add_septic_shock_flag(
        init_traj, shock_cfg=config["septic_shock"]
    )

    return init_traj
