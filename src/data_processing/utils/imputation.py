import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from scipy.interpolate import interp1d
from sklearn.impute import KNNImputer


def sample_and_hold(df, vitalslab_hold):
    print("Performing sample and hold interpolation...")

    # ensure correct ordering before forward-fill logic
    df = df.sort_values(["stay_id", "charttime"]).reset_index(drop=True)

    cols_to_process = [
        col
        for col in vitalslab_hold
        if col in df.columns and np.issubdtype(df[col].dtype, np.number)
    ]

    for col in tqdm(cols_to_process, desc="\tProcessing columns", ncols=100):
        hold_period_secs = vitalslab_hold[col] * 3600

        # for each row, record the charttime of the last valid (non-NaN) measurement
        # within the same stay, then only carry the value forward if it's within
        # the hold period
        last_valid_time = (
            df[["stay_id", "charttime", col]]
            .where(df[col].notna())  # NaN out rows where col is missing
            .groupby(df["stay_id"])["charttime"]
            .transform(lambda x: x.ffill())  # forward fill the charttime of last valid
        )

        last_valid_value = (
            df.groupby("stay_id")[col].transform(
                lambda x: x.ffill()
            )  # forward fill the value itself
        )

        # only apply the held value where:
        # 1. the original value is NaN (it's a gap)
        # 2. the last valid measurement is within the hold period
        within_hold = (df["charttime"] - last_valid_time) <= hold_period_secs
        gap_mask = df[col].isna()

        df.loc[gap_mask & within_hold, col] = last_valid_value[gap_mask & within_hold]

    return df


def fixgaps(x: np.ndarray) -> np.ndarray:
    """
    fix gaps in a 1D array using linear interpolation. Only fills NaNs that are
    between valid measurements, leaves leading and trailing NaNs as they are.
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


def handle_missing_values(df, missing_threshold=0.8):
    """
    Handle missing values by:
    1. Dropping columns with missingness above missing_threshold.
    2. Linear interpolation (per patient) for columns with low missingness (< 5%).
    3. KNN imputation for columns with moderate missingness, in patient-aligned
       chunks to avoid imputing across patient boundaries.
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

    # 1. drop columns with too many missing values
    miss = df[measurement_cols].isna().sum() / len(df)
    cols_to_keep = miss[miss < missing_threshold].index
    df = df[
        df.columns[~df.columns.isin(measurement_cols)].tolist() + cols_to_keep.tolist()
    ]

    # 2. linear interpolation per patient for low-missingness columns
    low_missing_cols = miss[(miss > 0) & (miss < 0.05)].index
    for col in tqdm(low_missing_cols, desc="\tLinear interpolation", ncols=100):
        if col in df.columns:
            df[col] = df.groupby("stay_id")[col].transform(
                lambda x: pd.Series(fixgaps(x.values), index=x.index)
            )

    # 3. KNN imputation for moderate-missingness columns
    cols_for_knn = [
        c for c in cols_to_keep if c not in low_missing_cols and c in df.columns
    ]

    if cols_for_knn:
        ref = df[cols_for_knn].values.copy()

        # build patient-aligned chunks so KNN never imputes across patient boundaries
        # each chunk contains whole patients and grows until it reaches chunk_size rows
        print("\tBuilding patient-aligned chunks for KNN...")
        patient_groups = df.groupby("stay_id", sort=False).apply(
            lambda x: x.index.tolist()
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
            imputer = KNNImputer(n_neighbors=1, keep_empty_features=True)
            ref[chunk_indices, :] = imputer.fit_transform(ref[chunk_indices, :])

        df[cols_for_knn] = ref
        print(f"\tRemaining missing values after KNN: {df.isna().sum().sum()}")

    return df


def add_missingness_features(
    df, lab_cols=["lactate", "wbc", "creatinine", "platelets"], timestep_hours=4
):
    """
    Captures the informative missingness of critical labs BEFORE they are imputed.
    Calculates whether a lab was measured in a timestep, and how many hours it has
    been since the last measurement.
    """
    print("Calculating informative missingness features...")

    # Ensure correct ordering
    df = df.sort_values(["stay_id", "timestep"])

    for col in lab_cols:
        if col not in df.columns:
            continue

        print(f"\tAdding missingness features for {col}...")

        # 1. Binary indicator: 1 if the data is NOT missing, 0 if NaN
        df[f"{col}_measured"] = df[col].notna().astype(float)

        # 2. Hours since last measurement
        # Find the specific timestep where it was measured
        last_measured_step = df["timestep"].where(df[f"{col}_measured"] == 1)

        # Forward fill the timestep per patient
        last_measured_step = df.groupby("stay_id")[last_measured_step.name].ffill()

        # Calculate hours
        df[f"hours_since_{col}"] = (
            df["timestep"] - last_measured_step
        ) * timestep_hours

        # Fill NaNs for the timesteps occurring before their very first lab was drawn
        df[f"hours_since_{col}"] = df[f"hours_since_{col}"].fillna(999.0)

    return df
