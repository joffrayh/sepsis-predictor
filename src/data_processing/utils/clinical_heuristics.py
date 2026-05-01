import json
import os

import numpy as np
from tqdm.auto import tqdm


def handle_outliers(df, config_path="src/data_processing/configs/outlier_bounds.json"):
    """
    Apply per-variable clinical bounding rules from a JSON config, then enforce
    hardcoded special-case rules for SpO\u2082, temperature, and FiO\u2082.

    Config-driven rules are read from ``outlier_bounds.json`` and applied in this
    order within each variable: (1) nullify values outside ``min_valid``/``max_valid``;
    (2) convert in-range Fahrenheit values to Celsius; (3) forward/back-fill sensor
    dropouts within each stay; (4) hard-clip to ``clip_low``/``clip_high``; (5) apply
    ``log1p`` transform if specified. All fields are optional — a missing field skips
    that rule for the variable.

    Hardcoded rules (applied after the config loop, always if the column is present):

    - **SpO\u2082**: values > 150 \u2192 NaN; values in (100, 150] \u2192 clipped to 100.
    - **temp_C**: values > 90 \u2192 NaN. If ``temp_F`` exists and is NaN at that row,
      the value is first rescued into ``temp_F`` (assumed Fahrenheit recorded in
      the wrong column).
    - **FiO\u2082**: values > 100 \u2192 NaN; values < 1 scaled \xd7100 (0\u20131 fraction fix);
      values < 20 \u2192 NaN (below room air).

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format measurement DataFrame (output of ``process_patient_measurements``).
        Values are raw, unprocessed measurement numbers.
    config_path : str, optional
        Path to ``outlier_bounds.json``. If the file does not exist, config-driven
        rules are skipped and a warning is printed; hardcoded rules still run.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with out-of-range values nullified, clipped, or transformed.
        Mutated in place and returned.

    Notes
    -----
    The SpO\u2082 rule is a strict two-step sequence: nullify > 150 first, then clip
    (100, 150] \u2192 100. Reversing the order would incorrectly clip values that should
    be NaN.

    The temperature rescue copies ``temp_C > 90`` into ``temp_F`` only where
    ``temp_F`` is NaN, to avoid overwriting a valid Fahrenheit reading.
    """
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

        # 2. In-column Fahrenheit values: convert to Celsius
        f_to_c_range = rules.get("convert_f_to_c")
        if f_to_c_range is not None:
            f_mask = (df[col] >= f_to_c_range[0]) & (df[col] <= f_to_c_range[1])
            if f_mask.any():
                df.loc[f_mask, col] = (df.loc[f_mask, col] - 32) / 1.8

        # 3. Recover sensor dropouts: forward/back-fill within each stay
        impute_extreme = rules.get("impute_extreme_nans")
        if impute_extreme:
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

    # SpO₂: nullify impossible values first, then clip plausible out-of-scale readings — order matters
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
    Impute missing GCS values from RASS scores using a fixed clinical mapping.

    GCS is required for SOFA CNS sub-scoring. When GCS is absent but RASS is
    documented — common in MIMIC-IV ICU stays — this provides a reasonable
    approximation: RASS 0-4 → GCS 15; RASS -1 → 14; -2 → 12; -3 → 11;
    -4 → 6; -5 → 3.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format measurement DataFrame. Must contain a ``richmond_ras`` column.
        A ``gcs`` column is created as all-NaN if not already present.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with ``gcs`` filled where it was NaN and a RASS mapping
        was available. Existing GCS values are never overwritten. RASS values
        outside the mapping table result in NaN. Mutated in place and returned.
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
    Estimate FiO₂ (as a percentage, 21-100) from oxygen flow and delivery device
    when the direct FiO₂ measurement is missing.

    Flow is resolved with priority: ``oxygen_flow`` → ``oxygen_flow_cannula_rate``
    → ``oxygen_flow_rate`` (first non-NaN wins). Device codes correspond to MIMIC-IV
    ``oxygen_flow_device`` chartevents values. Estimation uses per-device
    flow-to-FiO₂ lookup tables derived from standard clinical guidelines.

    Device handling:

    - Codes ``"0"``, ``"2"`` (nasal cannula/simple): 1-15 L/min → 24-70%; unknown
      flow → 21% (room air).
    - Codes ``"3"``-``"6"``, ``"8"``-``"12"`` (face mask variants): 4-15 L/min → 36-75%.
    - Code ``"7"`` (high-flow oxygen): ≤6-≥15 L/min → 60-100%.
    - Code ``"13"`` (high-flow face mask): <10-≥15 L/min → 60-100%.
    - Code ``"14"`` (simple face mask): <5-≥10 L/min → 40-80%.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format measurement DataFrame. Must have columns ``fio2``,
        ``oxygen_flow``, ``oxygen_flow_cannula_rate``, ``oxygen_flow_rate``,
        and ``oxygen_flow_device``.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with ``fio2`` filled where estimation was possible.
        Mutated in place and returned.

    Notes
    -----
    Lookup tables iterate from the highest threshold downward; each pass overwrites
    rows where flow ≤ threshold, so the smallest matching threshold wins — equivalent
    to a range lookup. Only rows where ``fio2`` is already NaN are eligible; no
    existing values are modified.
    """
    print("Estimating FiO2...")

    # Codes "1", "15" (Ultrasonic neb), "16" (Vapomist), and "17" (Other) are intentionally
    # unhandled as they are humidity/medication delivery devices, not supplemental O₂ sources,
    # so flow-based FiO₂ estimation does not apply. Code "1" is never produced by the
    # extraction query (the numbering skips from "0" to "2").

    # keep flow as a local Series — never write it into df to avoid block consolidation copies
    flow = (
        df["oxygen_flow"]
        .fillna(df["oxygen_flow_cannula_rate"])
        .fillna(df["oxygen_flow_rate"])
    )

    # Nasal cannula and simple flow devices (codes "0", "2")
    mask = (
        (df["fio2"].isna())
        & (flow.notna())
        & (df["oxygen_flow_device"].isin(["0", "2"]))
    )
    if mask.any():
        for threshold, fio2_val in zip(
            [15, 12, 10, 8, 6, 5, 4, 3, 2, 1],
            [70, 62, 55, 50, 44, 40, 36, 32, 28, 24],
        ):
            df.loc[mask & (flow <= threshold), "fio2"] = fio2_val

    # No flow recorded on a simple device → assume room air (21%)
    mask = (
        (df["fio2"].isna())
        & (flow.isna())
        & (df["oxygen_flow_device"].isin(["0", "2"]))
    )
    df.loc[mask, "fio2"] = 21

    # Face mask variants (codes "3"-"6", "8"-"12")
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

    # High-flow oxygen (code "7")
    mask = (df["fio2"].isna()) & (flow.notna()) & (df["oxygen_flow_device"] == "7")
    if mask.any():
        df.loc[mask & (flow >= 15), "fio2"] = 100
        df.loc[mask & (flow >= 10) & (flow < 15), "fio2"] = 90
        df.loc[mask & (flow < 10) & (flow > 8), "fio2"] = 80
        df.loc[mask & (flow <= 8) & (flow > 6), "fio2"] = 70
        df.loc[mask & (flow <= 6), "fio2"] = 60

    # High-flow face mask (code "13")
    mask = (df["fio2"].isna()) & (flow.notna()) & (df["oxygen_flow_device"] == "13")
    if mask.any():
        df.loc[mask & (flow >= 15), "fio2"] = 100
        df.loc[mask & (flow >= 10) & (flow < 15), "fio2"] = 80
        df.loc[mask & (flow < 10), "fio2"] = 60

    # Simple face mask (code "14")
    mask = (df["fio2"].isna()) & (flow.notna()) & (df["oxygen_flow_device"] == "14")
    if mask.any():
        df.loc[mask & (flow >= 10), "fio2"] = 80
        df.loc[mask & (flow >= 5) & (flow < 10), "fio2"] = 60
        df.loc[mask & (flow < 5), "fio2"] = 40

    return df


def handle_unit_conversions(df):
    """
    Harmonise temperature, haemoglobin/haematocrit, and bilirubin units by
    detecting cross-column mis-recordings and cross-filling from complementary
    measurements.

    Rules applied in order:

    1. ``temp_F ∈ (25, 45)`` → move to ``temp_C``, nullify ``temp_F``
       (Celsius value recorded in the Fahrenheit column).
    2. ``temp_C > 70`` → move to ``temp_F``, nullify ``temp_C``
       (Fahrenheit value recorded in the Celsius column).
    3. Fill ``temp_F`` from ``temp_C`` where ``temp_F`` is NaN: ``F = C x 1.8 + 32``.
    4. Fill ``temp_C`` from ``temp_F`` where ``temp_C`` is NaN: ``C = (F - 32) / 1.8``.
    5. Fill ``hematocrit`` from ``hemoglobin``: ``Hct = (Hgb x 2.862) + 1.216``.
    6. Fill ``hemoglobin`` from ``hematocrit``: ``Hgb = (Hct - 1.216) / 2.862``.
    7. Fill ``bilirubin_direct`` from ``bilirubin_total``:
       ``Direct = (Total x 0.6934) - 0.1752``.
    8. Fill ``bilirubin_total`` from ``bilirubin_direct``:
       ``Total = (Direct + 0.1752) / 0.6934``.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format measurement DataFrame. Must contain ``temp_C``, ``temp_F``,
        ``hemoglobin``, ``hematocrit``, ``bilirubin_total``, and
        ``bilirubin_direct`` columns.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with cross-filled missing values. No existing value is
        ever overwritten. Mutated in place and returned.

    Notes
    -----
    Rules 1-2 (rescue of cross-column mis-recordings) run before rules 3-4
    (forward filling), so rescued values can feed into the same-pass fill.
    The haemoglobin and bilirubin formulas are empirical linear approximations.
    """
    print("Converting units...")
    mask = (df["temp_F"] > 25) & (df["temp_F"] < 45)
    if mask.any():
        df.loc[mask, "temp_C"] = df.loc[mask, "temp_F"]
        df.loc[mask, "temp_F"] = np.nan

    mask = df["temp_C"] > 70
    if mask.any():
        df.loc[mask, "temp_F"] = df.loc[mask, "temp_C"]
        df.loc[mask, "temp_C"] = np.nan

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
