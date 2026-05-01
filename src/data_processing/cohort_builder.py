import os

import numpy as np
import pandas as pd


def process_microbio_data(microbio, culture):
    """
    Fuse microbiology and culture events into a single bacteriology table.

    Fills missing ``charttime`` from ``chartdate`` for rows where MIMIC-IV recorded
    only an event date rather than an exact timestamp, then drops ``chartdate``.

    Parameters
    ----------
    microbio : pd.DataFrame
        Raw microbiology events. Must have ``charttime`` and ``chartdate`` columns.
    culture : pd.DataFrame
        Raw culture events.

    Returns
    -------
    pd.DataFrame
        Concatenated bacteriology table with ``charttime`` filled where possible
        and ``chartdate`` dropped. Rows where both fields were NaN remain NaN.

    Notes
    -----
    Modifies the ``charttime`` column of the input ``microbio`` DataFrame in place
    before concatenation. The caller's copy of ``microbio`` will reflect this change.
    """
    microbio["charttime"] = microbio["charttime"].fillna(microbio["chartdate"])
    microbio = microbio.drop(columns=["chartdate"])
    return pd.concat([microbio, culture], sort=False, ignore_index=True)


def process_demog_data(demog):
    """
    Clean the demographics table by filling missing outcome and comorbidity fields
    and deduplicating on hospital admission window.

    Parameters
    ----------
    demog : pd.DataFrame
        Raw demographics from ``demog.csv``.

    Returns
    -------
    pd.DataFrame
        Cleaned copy, deduplicated on ``(admittime, dischtime)`` keeping the first row.

    Notes
    -----
    ``morta_90``, ``morta_hosp``, and ``charlson_comorbidity_index`` are filled with 0,
    not excluded — missing values are treated as no event or no comorbidity rather
    than unknown. Deduplication targets the hospital admission window, not ``stay_id``,
    because a single admission can span multiple ICU stays.
    """
    demog["morta_90"] = demog["morta_90"].fillna(0)
    demog["morta_hosp"] = demog["morta_hosp"].fillna(0)
    demog["charlson_comorbidity_index"] = demog["charlson_comorbidity_index"].fillna(0)
    return demog.drop_duplicates(subset=["admittime", "dischtime"], keep="first").copy()


def calculate_readmissions(demog, cutoff_days=30):
    """
    Add a binary ``re_admission`` flag to the demographics table.

    A stay is flagged as a readmission if the patient's previous ICU discharge
    (within the same ``subject_id``) occurred within ``cutoff_days`` days of the
    current admission.

    Parameters
    ----------
    demog : pd.DataFrame
        Cleaned demographics (output of ``process_demog_data``). Must have
        ``subject_id``, ``admittime``, and ``dischtime`` as numeric Unix timestamps.
    cutoff_days : int, optional
        Readmission window in days (default 30).

    Returns
    -------
    pd.DataFrame
        Input dataframe with ``re_admission`` column added (0/1) and the
        temporary ``prev_dischtime`` column dropped.

    Notes
    -----
    Uses a vectorised ``groupby + shift(1)`` rather than a per-subject loop.
    The first admission for each subject always gets ``re_admission = 0`` because
    ``prev_dischtime`` is NaN for the first row. The cutoff comparison is in
    seconds: ``cutoff = cutoff_days × 24 × 3600``.

    ``admittime`` and ``dischtime`` are normalised to float Unix seconds if they
    arrive as ``datetime64`` — the pipeline currently produces numeric timestamps,
    but this guards against future changes to the extraction query.
    """
    print("Calculating readmissions...")
    cutoff = cutoff_days * 24 * 3600

    demog = demog.sort_values(["subject_id", "admittime"])

    # Convert datetime64 columns to float Unix seconds so the seconds-based cutoff
    # comparison is valid regardless of how timestamps were loaded. NaT becomes NaN.
    for col in ("admittime", "dischtime"):
        if pd.api.types.is_datetime64_any_dtype(demog[col]):
            demog[col] = (demog[col] - pd.Timestamp("1970-01-01")) / pd.Timedelta("1s")

    demog["prev_dischtime"] = demog.groupby("subject_id")["dischtime"].shift(1)

    demog["re_admission"] = 0
    mask = (demog["admittime"] - demog["prev_dischtime"]) <= cutoff
    demog.loc[mask, "re_admission"] = 1

    return demog.drop(columns=["prev_dischtime"])


def fill_missing_icustay_ids(bacterio, demog, abx, stay_id_match_window_hours=48):
    """
    Assign ``stay_id`` to bacteriology and antibiotic events that lack one.

    An event is linked to a stay if its timestamp falls within
    ``stay_id_match_window_hours`` of the stay's ``intime``/``outtime``, or if the
    patient has exactly one ICU stay in the dataset (single-stay fallback).
    Bacteriology is matched on ``subject_id``; ABx on ``hadm_id``, reflecting the
    MIMIC-IV schema where antibiotic records are keyed to hospital admissions.

    Parameters
    ----------
    bacterio : pd.DataFrame
        Bacteriology events (output of ``process_microbio_data``). Rows with
        ``stay_id = NaN`` are the fill targets.
    demog : pd.DataFrame
        Processed demographics with ``subject_id``, ``hadm_id``, ``stay_id``,
        ``intime``, and ``outtime``.
    abx : pd.DataFrame
        Antibiotics table. Mutated in place; a temporary ``idx`` column is added
        and removed.
    stay_id_match_window_hours : int, optional
        Symmetric tolerance window around each stay boundary in hours (default 48).

    Returns
    -------
    tuple of pd.DataFrame
        ``(bacterio, abx)`` with ``stay_id`` filled where a match was found.
        Events with no matching stay retain ``stay_id = NaN`` and are filtered
        downstream.

    Notes
    -----
    When a missing-``stay_id`` event matches multiple stays within the window,
    the last matching stay wins (``drop_duplicates(keep="last")``). This is applied
    to both bacteriology and ABx events before the index-based assignment to ensure
    each source row maps to exactly one stay.
    """
    print("Filling-in missing ICUSTAY IDs in bacterio")

    missing_bact = bacterio[bacterio["stay_id"].isna()].copy()
    missing_bact["idx"] = missing_bact.index

    # drop stay_id before merge to avoid pandas creating stay_id_x / stay_id_y
    missing_bact = missing_bact.drop(columns=["stay_id"])

    demog_sub = (
        demog[["subject_id", "stay_id", "intime", "outtime"]]
        .dropna(subset=["subject_id"])
        .copy()
    )
    stay_counts = demog_sub.groupby("subject_id").size().reset_index(name="stay_count")
    demog_sub = demog_sub.merge(stay_counts, on="subject_id")

    merged_bact = missing_bact.merge(demog_sub, on="subject_id", how="inner")
    time_mask = (
        merged_bact["charttime"]
        >= merged_bact["intime"] - stay_id_match_window_hours * 3600
    ) & (
        merged_bact["charttime"]
        <= merged_bact["outtime"] + stay_id_match_window_hours * 3600
    )
    valid_bact = merged_bact[time_mask | (merged_bact["stay_count"] == 1)]

    # Deduplicate so each bacteriology row maps to at most one stay before
    # assignment — prevents positional misalignment when multiple stays matched.
    valid_bact = valid_bact.drop_duplicates(subset=["idx"], keep="last")
    bacterio.loc[valid_bact["idx"], "stay_id"] = valid_bact["stay_id"].values

    print("Filling-in missing ICUSTAY IDs in ABx")

    abx["idx"] = abx.index

    demog_abx_sub = (
        demog[["hadm_id", "stay_id", "intime", "outtime"]]
        .dropna(subset=["hadm_id"])
        .copy()
    )
    abx_counts = demog_abx_sub.groupby("hadm_id").size().reset_index(name="stay_count")
    demog_abx_sub = demog_abx_sub.merge(abx_counts, on="hadm_id")

    merged_abx = abx.merge(demog_abx_sub, on="hadm_id", how="inner")
    time_mask_abx = (
        merged_abx["starttime"]
        >= merged_abx["intime"] - stay_id_match_window_hours * 3600
    ) & (
        merged_abx["starttime"]
        <= merged_abx["outtime"] + stay_id_match_window_hours * 3600
    )
    valid_abx = merged_abx[time_mask_abx | (merged_abx["stay_count"] == 1)]

    # Deduplicate so each ABx event maps to at most one stay before assignment.
    valid_abx = valid_abx.drop_duplicates(subset=["idx"], keep="last")

    abx.loc[valid_abx["idx"], "stay_id"] = valid_abx["stay_id"].values
    abx = abx.drop(columns=["idx"])

    return bacterio, abx


def find_infection_onset(
    abx, bacterio, abx_before_culture_hours=24, abx_after_culture_hours=72
):
    """
    Identify presumed infection onset for each ICU stay using the Sepsis-3 definition.

    An onset is confirmed when an antibiotic administration and a culture sample are
    temporally proximate: ABx within ``abx_before_culture_hours`` hours *before* culture
    (onset = ABx ``starttime``), or culture within ``abx_after_culture_hours`` hours
    *before* ABx (onset = culture ``charttime``).

    Parameters
    ----------
    abx : pd.DataFrame
        Antibiotics table with ``stay_id`` and ``starttime``. Rows with
        ``stay_id = NaN`` are excluded.
    bacterio : pd.DataFrame
        Bacteriology table with ``stay_id``, ``subject_id``, and ``charttime``.
        Rows with ``stay_id = NaN`` are excluded.
    abx_before_culture_hours : int, optional
        Maximum hours an antibiotic may precede a culture and still qualify
        (Sepsis-3 criterion 1, default 24).
    abx_after_culture_hours : int, optional
        Maximum hours a culture may precede an antibiotic and still qualify
        (Sepsis-3 criterion 2, default 72).

    Returns
    -------
    pd.DataFrame
        Columns ``subject_id``, ``stay_id``, ``onset_time`` — one row per stay
        with the earliest valid onset. Stays with no qualifying ABx+culture pair
        are absent from this frame and receive ``onset_time = NaN`` in the cohort.

    Notes
    -----
    For each antibiotic event, only the *closest* culture in time is considered
    (deduplicated on ``abx_idx`` after sorting by ``diff_hr``). Among all valid
    onset candidates for a stay, only the earliest (lowest ``abx_idx``) is kept.

    Onset time is assigned via ``np.where(cond1, starttime, charttime)``. Because
    ``valid`` was already filtered to rows satisfying ``cond1 | cond2``, re-evaluating
    ``cond1`` on ``valid`` acts as a ternary: abx-before-culture rows get ``starttime``,
    all others get ``charttime``.
    """
    print("Finding presumed onset of infection according to sepsis3 guidelines")

    abx_sub = abx[["stay_id", "starttime"]].dropna(subset=["stay_id"]).copy()
    # Capture the original row index — used later to select the earliest onset
    # candidate per stay from among all qualifying ABx+culture pairs
    abx_sub["abx_idx"] = abx_sub.index

    bact_sub = (
        bacterio[["stay_id", "subject_id", "charttime"]]
        .dropna(subset=["stay_id"])
        .copy()
    )

    merged = pd.merge(abx_sub, bact_sub, on="stay_id", how="inner")

    merged["diff_hr"] = (merged["starttime"] - merged["charttime"]).abs() / 3600
    merged = merged.sort_values(["abx_idx", "diff_hr"])
    closest = merged.drop_duplicates(subset=["abx_idx"]).copy()

    cond1 = (closest["diff_hr"] <= abx_before_culture_hours) & (
        closest["starttime"] <= closest["charttime"]
    )
    cond2 = (closest["diff_hr"] <= abx_after_culture_hours) & (
        closest["starttime"] >= closest["charttime"]
    )

    valid = closest[cond1 | cond2].copy()

    valid_cond1 = (valid["diff_hr"] <= abx_before_culture_hours) & (
        valid["starttime"] <= valid["charttime"]
    )

    # np.where acts as a ternary: valid already satisfies cond1|cond2, so
    # re-evaluating cond1 here assigns starttime (abx-before-culture) or charttime
    valid["onset_time"] = np.where(valid_cond1, valid["starttime"], valid["charttime"])

    valid = valid.sort_values("abx_idx").drop_duplicates(
        subset=["stay_id"], keep="first"
    )

    onset_df = valid[["subject_id", "stay_id", "onset_time"]].copy()
    print(f"Number of preliminary, presumed septic trajectories: {len(onset_df)}")
    return onset_df


def build_full_cohort(onset_df, demog):
    """
    Merge presumed infection onset times onto the full ICU stay list.

    All stays are included; non-septic stays receive ``onset_time = NaN``.
    ICU ``intime`` is used as ``anchor_time`` for every stay, providing a
    consistent temporal reference for Phase 3 grid alignment regardless of
    whether infection was confirmed.

    Parameters
    ----------
    onset_df : pd.DataFrame
        Output of ``find_infection_onset``. Columns: ``subject_id``, ``stay_id``,
        ``onset_time``.
    demog : pd.DataFrame
        Processed demographics with ``subject_id``, ``stay_id``, and ``intime``.

    Returns
    -------
    pd.DataFrame
        One row per unique ICU stay with columns ``subject_id``, ``stay_id``,
        ``anchor_time``, ``intime``, and ``onset_time``.

    Notes
    -----
    ``anchor_time`` equals ``intime`` for all stays. Using admission time as the
    anchor (rather than onset time) ensures Phase 3 time windows are consistent
    across the full cohort. ``onset_time`` is retained separately for downstream
    labelling in Phase 3.
    """
    all_stays = demog[["subject_id", "stay_id", "intime"]].drop_duplicates()

    merged = all_stays.merge(onset_df, on=["subject_id", "stay_id"], how="left")

    # anchor is always intime — consistent across septic and non-septic patients
    merged["anchor_time"] = merged["intime"]

    # retain onset_time for downstream labelling
    return merged[["subject_id", "stay_id", "anchor_time", "intime", "onset_time"]]


def build_and_save_cohorts(config, path_config):
    """
    Orchestrate Phase 2 in full: load, process, and persist the patient cohort.

    Reads pipe-delimited CSVs from ``extracted_dir``, runs all cohort-building
    sub-steps in order, writes five output files to ``processed_dir``, and returns
    the processed DataFrames for immediate use by Phase 3.

    Parameters
    ----------
    config : dict
        The ``cohort`` section of ``config.yaml``. Required keys:
        ``readmission_window_days``, ``stay_id_match_window_hours``,
        ``infection_abx_before_culture_hours``, ``infection_abx_after_culture_hours``.
    path_config : dict
        The ``paths`` section of ``config.yaml``. Required keys:
        ``extracted_dir`` (input CSVs) and ``processed_dir`` (output destination).

    Returns
    -------
    tuple
        ``(cohort, bacterio, demog, data)`` where ``data`` is a dict with keys
        ``"labU"`` (unified lab frame) and ``"abx"`` (processed antibiotics table).

    Notes
    -----
    Input files (pipe-delimited, from ``extracted_dir``):
    ``abx.csv``, ``culture.csv``, ``microbio.csv``, ``demog.csv``,
    ``labs_ce.csv``, ``labs_le.csv``.

    Output files (pipe-delimited, to ``processed_dir``):
    ``cohort.csv``, ``bacterio_processed.csv``, ``demog_processed.csv``,
    ``labu.csv``, ``abx_processed.csv``.

    ``labs_le.csv`` uses ``timestp`` as its timestamp column; it is renamed to
    ``charttime`` before concatenation so the unified lab frame has a consistent
    column name throughout Phase 3.
    """
    input_dir = path_config["extracted_dir"]
    output_dir = path_config["processed_dir"]

    print("Loading required processed files...")
    files = {
        "abx": "abx.csv",
        "culture": "culture.csv",
        "microbio": "microbio.csv",
        "demog": "demog.csv",
    }

    data = {}
    for key, filename in files.items():
        data[key] = pd.read_csv(os.path.join(input_dir, filename), sep="|", low_memory=False)

    print("Loading and combining lab data...")
    labs_ce = pd.read_csv(os.path.join(input_dir, "labs_ce.csv"), sep="|", low_memory=False)
    labs_le = pd.read_csv(os.path.join(input_dir, "labs_le.csv"), sep="|", low_memory=False)
    labs_le = labs_le.rename(columns={"timestp": "charttime"})
    data["labU"] = pd.concat([labs_ce, labs_le], sort=False, ignore_index=True)

    bacterio = process_microbio_data(data["microbio"], data["culture"])
    demog = process_demog_data(data["demog"])
    demog = calculate_readmissions(demog, cutoff_days=config["readmission_window_days"])
    bacterio, data["abx"] = fill_missing_icustay_ids(
        bacterio,
        demog,
        data["abx"],
        stay_id_match_window_hours=config["stay_id_match_window_hours"],
    )
    onset = find_infection_onset(
        data["abx"],
        bacterio,
        abx_before_culture_hours=config["infection_abx_before_culture_hours"],
        abx_after_culture_hours=config["infection_abx_after_culture_hours"],
    )
    cohort = build_full_cohort(onset, demog)

    print("Saving processed files...")
    os.makedirs(output_dir, exist_ok=True)
    cohort.to_csv(os.path.join(output_dir, "cohort.csv"), sep="|", index=False)
    bacterio.to_csv(os.path.join(output_dir, "bacterio_processed.csv"), sep="|", index=False)
    demog.to_csv(os.path.join(output_dir, "demog_processed.csv"), sep="|", index=False)
    data["labU"].to_csv(os.path.join(output_dir, "labu.csv"), sep="|", index=False)
    data["abx"].to_csv(os.path.join(output_dir, "abx_processed.csv"), sep="|", index=False)

    print("Cohort building complete!")
    return cohort, bacterio, demog, data
