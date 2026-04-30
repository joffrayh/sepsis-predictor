import json
import os

import numpy as np
from tqdm.auto import tqdm


def handle_outliers(df, config_path="src/data_processing/configs/cleaning_config.json"):
    print("Handling outliers in patient timeseries data via dynamic config")

    if not os.path.exists(config_path):
        print(
            f"Warning: Config not found at {config_path}. Skipping dynamic outlier handling."
        )
        return df

    with open(config_path) as f:
        outlier_bounds = json.load(f)

    for col, rules in tqdm(
        outlier_bounds.items(),
        desc="	Applying config-driven clinical bounding",
        ncols=100,
    ):
        if col not in df.columns:
            continue

        # 1. Nullify absolute impossibilities
        min_val = rules.get("min_valid")
        if min_val is not None:
            df.loc[df[col] < min_val, col] = np.nan

        max_val = rules.get("max_valid")
        if max_val is not None:
            df.loc[df[col] > max_val, col] = np.nan

        # 2. Fahrenheit to Celsius conversions (Specific to Notebook logic)
        f_to_c_range = rules.get("convert_f_to_c")
        if f_to_c_range is not None:
            f_mask = (df[col] >= f_to_c_range[0]) & (df[col] <= f_to_c_range[1])
            if f_mask.any():
                df.loc[f_mask, col] = (df.loc[f_mask, col] - 32) / 1.8

        # 3. Handle Extreme NaNs (Specific to Notebook logic for Vitals dropping to 0)
        impute_extreme = rules.get("impute_extreme_nans")
        if impute_extreme:
            # We forward fill and backfill per stay_id
            df[col] = df.groupby("stay_id")[col].ffill()
            df[col] = df.groupby("stay_id")[col].bfill()

        # 4. Hard Clipping (Clinical domain bounds)
        clip_low = rules.get("clip_low")
        if clip_low is not None:
            df.loc[df[col] < clip_low, col] = clip_low

        clip_high = rules.get("clip_high")
        if clip_high is not None:
            df.loc[df[col] > clip_high, col] = clip_high

        # 5. Transformations
        transform = rules.get("transform")
        if transform == "log1p":
            df[col] = np.log1p(df[col])

    # SpO2: two-step rule matching old pipeline
    if "spo2" in df.columns:
        df.loc[df["spo2"] > 150, "spo2"] = np.nan
        df.loc[df["spo2"] > 100, "spo2"] = 100

    # temp_C: values > 90 are assumed to be in Fahrenheit; rescue into temp_F then nullify
    if "temp_C" in df.columns:
        if "temp_F" in df.columns:
            mask = (df["temp_C"] > 90) & (df["temp_F"].isna())
            df.loc[mask, "temp_F"] = df.loc[mask, "temp_C"]
        df.loc[df["temp_C"] > 90, "temp_C"] = np.nan

    # FiO2: normalise 0-1 scale to percent, then apply plausibility bounds
    if "fio2" in df.columns:
        df.loc[df["fio2"] > 100, "fio2"] = np.nan
        df.loc[df["fio2"] < 1, "fio2"] *= 100
        df.loc[df["fio2"] < 20, "fio2"] = np.nan

    return df


def estimate_gcs_from_rass(df):
    """
    Estimate GCS (Glascow Coma Scale. which measures level of consciousness)
    from RASS (Richmond Agitation-Sedation Scale, which measures level of
    agitation or sedation) when GCS is missing, based on certain mappings.
    """
    print("Estimating GCS from RASS when GCS is missing...")
    if "gcs" not in df.columns:
        print("\tGCS column not found in dataframe.")
        print("\tSetting it to NaN and estimating from RASS where possible.")
        df["gcs"] = np.nan

    mappings = {0: 15, 1: 15, 2: 15, 3: 15, 4: 15, -1: 14, -2: 12, -3: 11, -4: 6, -5: 3}

    mask = df["gcs"].isna()
    df.loc[mask, "gcs"] = df.loc[mask, "richmond_ras"].map(mappings)

    return df


def estimate_fio2(df):
    """
    Estimate FiO2 (fraction of inspired oxygen) when missing, based on oxygen
    flow rates and device types.
    """
    print("Estimating FiO2...")

    # keep flow as a local Series — never write it into df to avoid block consolidation copies
    flow = df["oxygen_flow"].fillna(df["oxygen_flow_cannula_rate"]).fillna(df["oxygen_flow_rate"])

    # for simple flow devices (e.g. nasal cannula), use these flow-to-FiO2 mappings based on clinical estimates
    mask = (
        (df["fio2"].isna())
        & (flow.notna())
        & (df["oxygen_flow_device"].isin(["0", "2"]))
    )
    if mask.any():
        # iterate high-to-low: each pass overwrites rows ≤ threshold, so the
        # last match (smallest threshold ≥ flow) wins — same logic as original
        for threshold, fio2_val in zip(
            [15, 12, 10, 8, 6, 5, 4, 3, 2, 1],
            [70, 62, 55, 50, 44, 40, 36, 32, 28, 24],
        ):
            df.loc[mask & (flow <= threshold), "fio2"] = fio2_val

    # for patients with missing FiO2 but no flow recorded,
    # if device is simple (e.g. nasal cannula) we can assume room air (21% FiO2)
    mask = (
        (df["fio2"].isna())
        & (flow.isna())
        & (df["oxygen_flow_device"].isin(["0", "2"]))
    )
    df.loc[mask, "fio2"] = 21

    # for face mask devices, use these flow-to-FiO2 mappings based on clinical estimates
    face_mask_types = ["3", "4", "5", "6", "8", "9", "10", "11", "12"]
    mask = (
        (df["fio2"].isna())
        & (flow.notna())
        & (df["oxygen_flow_device"].isin(face_mask_types))
    )
    if mask.any():
        for threshold, fio2_val in zip(
            [15, 12, 10, 8, 6, 4],
            [75, 69, 66, 58, 40, 36],
        ):
            df.loc[mask & (flow <= threshold), "fio2"] = fio2_val

    # for high flow devices, use these flow-to-FiO2 mappings based on clinical estimates
    mask = (
        (df["fio2"].isna())
        & (flow.notna())
        & (df["oxygen_flow_device"] == "7")
    )
    if mask.any():
        df.loc[mask & (flow >= 15), "fio2"] = 100
        df.loc[mask & (flow >= 10) & (flow < 15), "fio2"] = 90
        df.loc[mask & (flow < 10) & (flow > 8), "fio2"] = 80
        df.loc[mask & (flow <= 8) & (flow > 6), "fio2"] = 70
        df.loc[mask & (flow <= 6), "fio2"] = 60

    # for high flow face mask devices, use these flow-to-FiO2 mappings based on clinical estimates
    mask = (
        (df["fio2"].isna())
        & (flow.notna())
        & (df["oxygen_flow_device"] == "13")
    )
    if mask.any():
        df.loc[mask & (flow >= 15), "fio2"] = 100
        df.loc[mask & (flow >= 10) & (flow < 15), "fio2"] = 80
        df.loc[mask & (flow < 10), "fio2"] = 60

    # for simple face mask devices, use these flow-to-FiO2 mappings based on clinical estimates
    mask = (
        (df["fio2"].isna())
        & (flow.notna())
        & (df["oxygen_flow_device"] == "14")
    )
    if mask.any():
        df.loc[mask & (flow >= 10), "fio2"] = 80
        df.loc[mask & (flow >= 5) & (flow < 10), "fio2"] = 60
        df.loc[mask & (flow < 5), "fio2"] = 40

    return df


def handle_unit_conversions(df):
    """
    Handle unit conversions for temperature, hemoglobin/hematocrit and bilirubin.

    The function uses empirical conversion formulas to fill in missing values
    when one measurement is available but its corresponding unit/related measurement is not.

    - Temperature conversions use standard F-C formula: C = (F - 32) / 1.8
    - Hemoglobin-hematocrit relationship: Hct = (Hgb * 2.862) + 1.216
    - Bilirubin relationship: Dir = (Total * 0.6934) - 0.1752
    """
    print("Converting units...")
    mask = (df["temp_F"] > 25) & (df["temp_F"] < 45)
    if mask.any():
        df.loc[mask, "temp_C"] = df.loc[mask, "temp_F"]
        df.loc[mask, "temp_F"] = None

    mask = df["temp_C"] > 70
    if mask.any():
        df.loc[mask, "temp_F"] = df.loc[mask, "temp_C"]
        df.loc[mask, "temp_C"] = None

    mask = (~df["temp_C"].isna()) & (df["temp_F"].isna())
    if mask.any():
        df.loc[mask, "temp_F"] = df.loc[mask, "temp_C"] * 1.8 + 32

    mask = (~df["temp_F"].isna()) & (df["temp_C"].isna())
    if mask.any():
        df.loc[mask, "temp_C"] = (df.loc[mask, "temp_F"] - 32) / 1.8

    mask = (~df["hemoglobin"].isna()) & (df["hematocrit"].isna())
    if mask.any():
        df.loc[mask, "hematocrit"] = (df.loc[mask, "hemoglobin"] * 2.862) + 1.216

    mask = (~df["hematocrit"].isna()) & (df["hemoglobin"].isna())
    if mask.any():
        df.loc[mask, "hemoglobin"] = (df.loc[mask, "hematocrit"] - 1.216) / 2.862

    mask = (~df["bilirubin_total"].isna()) & (df["bilirubin_direct"].isna())
    if mask.any():
        df.loc[mask, "bilirubin_direct"] = (
            df.loc[mask, "bilirubin_total"] * 0.6934
        ) - 0.1752

    mask = (~df["bilirubin_direct"].isna()) & (df["bilirubin_total"].isna())
    if mask.any():
        df.loc[mask, "bilirubin_total"] = (
            df.loc[mask, "bilirubin_direct"] + 0.1752
        ) / 0.6934

    return df
