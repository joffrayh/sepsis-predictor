import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.impute import KNNImputer
from tqdm.auto import tqdm


def sample_and_hold(df, vitalslab_hold):
    """
    Forward-fill clinical measurements within each ICU stay up to a per-concept
    maximum hold time.

    Simulates the clinical reality that a measurement remains valid until a new
    one is taken or until it becomes too stale to trust. Applied on the raw wide
    table before grid standardisation so hold logic operates at the original
    measurement resolution, not on 4-hour averages.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format DataFrame sorted by ``(stay_id, charttime)`` — ordering
        guaranteed by ``process_patient_measurements``. Columns are concept names.
    vitalslab_hold : dict[str, int]
        Mapping from concept name to maximum hold time in hours. Only concepts
        present in both this dict and ``df.columns`` with a numeric dtype are
        processed. Source: ``hold_times`` returned by ``load_measurement_mappings``.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with NaN gaps filled where the gap falls within the hold
        period. Mutated in place and returned.

    Notes
    -----
    Two ``groupby(...).transform("ffill")`` passes are used: one to propagate the
    timestamp of the last valid reading, one to propagate its value. A gap is
    filled only when the current row is NaN **and** the elapsed time since the
    last valid reading is within the hold period. Gaps exceeding the hold period
    remain NaN for KNN to handle downstream.
    """
    print("Performing sample and hold interpolation...")

    cols_to_process = [
        col
        for col in vitalslab_hold
        if col in df.columns and np.issubdtype(df[col].dtype, np.number)
    ]

    for col in tqdm(cols_to_process, desc="\tProcessing columns", ncols=100):
        hold_period_secs = vitalslab_hold[col] * 3600

        last_valid_time = (
            df["charttime"]
            .where(df[col].notna())
            .groupby(df["stay_id"])
            .transform("ffill")
        )

        last_valid_value = df.groupby("stay_id")[col].transform("ffill")
        within_hold = (df["charttime"] - last_valid_time) <= hold_period_secs
        gap_mask = df[col].isna()

        df.loc[gap_mask & within_hold, col] = last_valid_value[gap_mask & within_hold]

    return df


def fixgaps(x: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate internal NaN gaps in a 1D array.

    Only fills NaNs that lie between two valid values; leading and trailing
    NaN runs are left untouched.

    Parameters
    ----------
    x : np.ndarray
        1D numeric array, may contain NaN.

    Returns
    -------
    np.ndarray
        Copy of ``x`` with internal gaps linearly interpolated. If ``x``
        contains no valid values, the unchanged copy is returned.
    """
    y = np.copy(x)
    nan_mask = np.isnan(x)
    valid_indices = np.arange(len(x))[~nan_mask]

    if len(valid_indices) == 0:
        return y

    nan_mask[: valid_indices[0]] = False
    nan_mask[valid_indices[-1] + 1 :] = False

    y[nan_mask] = interp1d(valid_indices, x[valid_indices])(np.arange(len(x))[nan_mask])
    return y


def handle_missing_values(df, missing_threshold=0.8, knn_neighbors=1):
    """
    Three-stage residual imputation on the standardised 4-hour trajectory grid.

    Operates on measurement columns only; metadata, treatment, and label columns
    are excluded throughout. Stages run in order:

    1. **Drop** — columns with global missingness \u2265 ``missing_threshold`` are
       removed entirely.
    2. **Linear interpolation** — per-patient internal gap filling for columns
       with 0% < missingness < 5%, using ``fixgaps``.
    3. **KNN** — ``KNNImputer`` on all remaining measurement columns with
       missingness \u2265 5%. Patients are grouped into chunks of ~9,999 rows (whole
       patients only) so KNN never imputes across patient boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory DataFrame (output of ``standardise_patient_trajectories``).
        Re-sorted by ``(stay_id, timestamp)`` internally.
    missing_threshold : float, optional
        Fraction above which a column is dropped entirely (default 0.8).
        Missingness is computed globally across all rows, not per-patient.
        Columns at exactly this value are kept (comparison is ``< missing_threshold``).
    knn_neighbors : int, optional
        Number of neighbours for ``KNNImputer`` (default 1).

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed measurement columns and high-missingness columns
        removed. Column order is otherwise preserved.

    Notes
    -----
    ``keep_empty_features=True`` is passed to ``KNNImputer`` so all-NaN columns
    within a chunk do not raise an error.

    Patient-chunk boundaries are determined by ``groupby("stay_id", sort=False)``
    (insertion order, not ``stay_id`` sort order) to avoid a redundant re-sort.
    """
    print("Handling missing values...")

    # ensure correct ordering before any imputation
    df = df.sort_values(["stay_id", "timestamp"]).reset_index(drop=True)

    # columns to exclude from missingness analysis — these are labels,
    # metadata, or treatment variables that should never be imputed
    non_measurement_cols = [
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
    measurement_cols = [col for col in df.columns if col not in non_measurement_cols]

    miss = df[measurement_cols].isna().sum() / len(df)
    cols_to_keep = miss[miss < missing_threshold].index
    df = df[
        df.columns[~df.columns.isin(measurement_cols)].tolist() + cols_to_keep.tolist()
    ]

    low_missing_cols = miss[(miss > 0) & (miss < 0.05)].index
    for col in tqdm(low_missing_cols, desc="\tLinear interpolation", ncols=100):
        if col in df.columns:
            df[col] = df.groupby("stay_id")[col].transform(
                lambda x: pd.Series(fixgaps(x.values), index=x.index)
            )

    cols_for_knn = [
        c for c in cols_to_keep if c not in low_missing_cols and c in df.columns
    ]

    if cols_for_knn:
        ref = df[cols_for_knn].values.copy()

        # Build patient-aligned chunks: accumulate whole patients until chunk_size
        # is reached so KNN never sees rows from two different patients in one fit
        print("\tBuilding patient-aligned chunks for KNN...")
        patient_groups = df.groupby("stay_id", sort=False).apply(
            lambda x: x.index.tolist(), include_groups=False
        )
        chunks = []
        current_chunk = []
        chunk_size = 9999
        for stay_indices in patient_groups:
            current_chunk.extend(stay_indices)
            if len(current_chunk) >= chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
        if current_chunk:
            chunks.append(current_chunk)

        print(
            f"\tRunning KNN imputation across {len(chunks)} patient-aligned chunks..."
        )
        for chunk_indices in tqdm(chunks, desc="\tKNN imputation", ncols=100):
            imputer = KNNImputer(n_neighbors=knn_neighbors, keep_empty_features=True)
            ref[chunk_indices, :] = imputer.fit_transform(ref[chunk_indices, :])

        df[cols_for_knn] = ref
        print(f"\tRemaining missing values after KNN: {df.isna().sum().sum()}")

    return df


def add_missingness_features(df, lab_cols=None, timestep_hours=4):
    """
    Add binary measurement indicators and time-since-last-measurement features
    for critical labs, capturing informative missingness before imputation.

    For each lab in ``lab_cols`` (if the column is present), two columns are appended:

    - ``<lab>_measured`` \u2014 float 0/1: 1 if the lab had a real value at this timestep.
    - ``hours_since_<lab>`` \u2014 float: hours since the last real measurement within
      the same patient. Pre-first-measurement rows receive 999.0 (sentinel meaning
      "never yet measured").

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory DataFrame (output of ``standardise_patient_trajectories``).
        Must have ``stay_id`` and ``timestep`` columns.
    lab_cols : list of str, optional
        Labs to generate features for. Defaults to the four SOFA-critical labs:
        ``["lactate", "wbc", "creatinine", "platelets"]``.
    timestep_hours : float, optional
        Duration of each timestep bin in hours (default 4). Must match the grid
        resolution used in ``standardise_patient_trajectories``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two new columns per lab, sorted by
        ``(stay_id, timestep)``. Labs absent from ``df.columns`` are silently
        skipped.

    Notes
    -----
    **Order-critical**: must be called after ``standardise_patient_trajectories``
    (to have the ``timestep`` column) and **before** ``handle_missing_values``.
    After KNN imputation, NaN patterns are destroyed and these features would be
    meaningless.

    ``hours_since_<lab>`` is computed in grid-time: the timestep index of the
    last valid reading is forward-filled within each patient, and the difference
    from the current index is multiplied by ``timestep_hours``. The 999.0 sentinel
    is deliberate \u2014 not NaN \u2014 so downstream models treat it as "very long ago"
    without special handling.
    """
    print("Calculating informative missingness features...")

    if lab_cols is None:
        lab_cols = ["lactate", "wbc", "creatinine", "platelets"]

    # Ensure correct ordering
    df = df.sort_values(["stay_id", "timestep"])

    for col in lab_cols:
        if col not in df.columns:
            continue

        print(f"\tAdding missingness features for {col}...")

        df[f"{col}_measured"] = df[col].notna().astype(float)

        last_measured_step = df["timestep"].where(df[f"{col}_measured"] == 1)
        # Forward-fill the timestep index of the last valid reading within each patient
        last_measured_step = df.groupby("stay_id")[last_measured_step.name].ffill()

        df[f"hours_since_{col}"] = (
            df["timestep"] - last_measured_step
        ) * timestep_hours

        # Rows before the patient's first measurement get the 999.0 sentinel
        df[f"hours_since_{col}"] = df[f"hours_since_{col}"].fillna(999.0)

    return df
