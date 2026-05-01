import numpy as np
import pandas as pd


def calculate_derived_variables(df):
    """
    Compute derived clinical features and scores from imputed measurements.

    Must run **after** KNN imputation — SOFA sub-scores and SIRS criteria
    require non-NaN values for ``pf_ratio``, ``platelets``, ``bilirubin_total``,
    ``map``, ``gcs``, ``creatinine``, ``uo_step``, ``temp_C``, ``heart_rate``,
    ``respiratory_rate``, ``arterial_co2_pressure``, and ``wbc``.

    Also applies several preprocessing corrections before scoring:
    ``gender`` shift, ``age`` de-identification cap, ``mechvent`` binarisation,
    ``charlson_comorbidity_index`` median fill, and zero-fill for vasopressor columns.

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory DataFrame after KNN imputation. Must contain all clinical
        measurement columns required for SOFA and SIRS scoring.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with the following columns added: ``pf_ratio``,
        ``shock_index``, ``sofa_resp``, ``sofa_coag``, ``sofa_liver``,
        ``sofa_cv``, ``sofa_cns``, ``sofa_renal``, ``sofa_score``,
        ``sirs_score``. Mutated in place and returned.

    Notes
    -----
    SOFA sub-scores use ``pd.cut`` with ``right=False`` (left-inclusive bins)
    and are filled with 0 where inputs are NaN, per the Sepsis-3 scoring convention.

    SOFA cardiovascular uses ``np.select`` with a priority order: missing data
    \u2192 0; MAP \u2265 70 \u2192 0; MAP 65\u201370 \u2192 1; MAP < 65 \u2192 2; ``vaso_max`` \u2264 0.1 \u2192 3;
    ``vaso_max`` > 0.1 \u2192 4.

    SOFA renal uses creatinine when available; falls back to ``uo_step`` (the
    4-hour urine output sum) only when creatinine is NaN.

    ``shock_index`` infinite values (zero SBP) are nullified then filled with
    the column mean.
    """
    # MIMIC-IV encodes gender as 1=Female/2=Male; shift to 0/1
    df["gender"] = df["gender"] - 1
    # MIMIC-IV de-identification caps age at 91 for patients ≥ 91 years old;
    # some rows encode this as a large integer rather than 91
    df.loc[df["age"] > 150, "age"] = 91.4
    df["mechvent"] = df["mechvent"].fillna(0)
    df.loc[df["mechvent"] > 0, "mechvent"] = 1
    df["charlson_comorbidity_index"] = df["charlson_comorbidity_index"].fillna(
        df["charlson_comorbidity_index"].median()
    )
    df["vaso_median"] = df["vaso_median"].fillna(0)
    df["vaso_max"] = df["vaso_max"].fillna(0)

    df["pf_ratio"] = df["arterial_o2_pressure"] / (df["fio2"] / 100)
    df["shock_index"] = df["heart_rate"] / df["sbp_arterial"]
    df.loc[np.isinf(df["shock_index"]), "shock_index"] = np.nan
    df["shock_index"] = df["shock_index"].fillna(df["shock_index"].mean())

    # SOFA respiratory
    pf = df["pf_ratio"]
    df["sofa_resp"] = (
        pd.cut(
            pf,
            bins=[-np.inf, 100, 200, 300, 400, np.inf],
            labels=[4, 3, 2, 1, 0],
            right=False,
        )
        .astype(float)
        .fillna(0)
        .astype(int)
    )

    # SOFA coagulation
    plt = df["platelets"]
    df["sofa_coag"] = (
        pd.cut(
            plt,
            bins=[-np.inf, 20, 50, 100, 150, np.inf],
            labels=[4, 3, 2, 1, 0],
            right=False,
        )
        .astype(float)
        .fillna(0)
        .astype(int)
    )

    # SOFA liver
    bili = df["bilirubin_total"]
    df["sofa_liver"] = (
        pd.cut(
            bili,
            bins=[-np.inf, 1.2, 2.0, 6.0, 12.0, np.inf],
            labels=[0, 1, 2, 3, 4],
            right=False,
        )
        .astype(float)
        .fillna(0)
        .astype(int)
    )

    # SOFA cardiovascular — np.select priority order matters: missing-data case
    # first, then MAP thresholds, then vasopressor dose
    map_ = df["map"]
    vaso = df["vaso_max"]
    cv_conditions = [
        map_.isna() & vaso.isna(),
        map_ >= 70,
        (map_ >= 65) & (map_ < 70),
        map_ < 65,
        vaso <= 0.1,
        vaso > 0.1,
    ]
    df["sofa_cv"] = np.select(cv_conditions, [0, 0, 1, 2, 3, 4], default=0)

    # SOFA CNS
    gcs = df["gcs"]
    df["sofa_cns"] = (
        pd.cut(
            gcs,
            bins=[-np.inf, 6, 9, 12, 14, np.inf],
            labels=[4, 3, 2, 1, 0],
            right=False,
        )
        .astype(float)
        .fillna(0)
        .astype(int)
    )

    # SOFA renal — creatinine takes priority; uo_step (4 h urine sum) is the fallback
    cr = df["creatinine"]
    uo = df["uo_step"]
    cr_score = pd.cut(
        cr,
        bins=[-np.inf, 1.2, 2.0, 3.5, 5.0, np.inf],
        labels=[0, 1, 2, 3, 4],
        right=False,
    ).astype(float)
    uo_score = np.select([uo >= 84, uo >= 34, uo < 34], [0, 3, 4], default=0)
    df["sofa_renal"] = np.where(
        cr.notna(), cr_score.fillna(0), np.where(uo.notna(), uo_score, 0)
    ).astype(int)

    df["sofa_score"] = (
        df["sofa_resp"]
        + df["sofa_coag"]
        + df["sofa_liver"]
        + df["sofa_cv"]
        + df["sofa_cns"]
        + df["sofa_renal"]
    )

    # SIRS
    sirs = pd.DataFrame(index=df.index)
    sirs["temp"] = ((df["temp_C"] >= 38) | (df["temp_C"] <= 36)).astype(int)
    sirs["hr"] = (df["heart_rate"] > 90).astype(int)
    sirs["rr"] = (
        (df["respiratory_rate"] >= 20) | (df["arterial_co2_pressure"] <= 32)
    ).astype(int)
    sirs["wbc"] = ((df["wbc"] >= 12) | (df["wbc"] < 4)).astype(int)
    df["sirs_score"] = sirs.sum(axis=1)
    del sirs

    return df


def apply_exclusion_criteria(df, exclusion_cfg):
    """
    Remove entire ICU stays that meet any of three physiological plausibility
    exclusion criteria.

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory DataFrame with ``stay_id``, ``uo_step``, ``fluid_step``,
        ``morta_hosp``, and ``timestamp`` columns.
    exclusion_cfg : dict
        Exclusion thresholds. Required keys:

        - ``max_uo_per_window_ml`` \u2014 UO ceiling per 4-hour window (mL).
        - ``max_fluid_per_window_ml`` \u2014 fluid ceiling per 4-hour window (mL).
        - ``early_death_hours`` \u2014 maximum stay duration (hours) for hospital
          deaths to be considered insufficiently informative.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with all timesteps for excluded stays removed.

    Notes
    -----
    Criteria are applied sequentially on the already-filtered result: exclusion
    counts reflect the population *after* prior criteria have been applied, not
    the original cohort size.

    Early death is computed from the ``timestamp`` span (Unix seconds) of the
    stay, not from a mortality date column.
    """
    max_uo = exclusion_cfg["max_uo_per_window_ml"]
    max_fluid = exclusion_cfg["max_fluid_per_window_ml"]
    early_death_hours = exclusion_cfg["early_death_hours"]

    print("Applying exclusion criteria...")
    excluded_counts = {}

    # Extreme UO
    extreme_uo_stays = df[df["uo_step"] > max_uo]["stay_id"].unique()
    df = df[~df["stay_id"].isin(extreme_uo_stays)]
    excluded_counts["extreme_uo"] = len(extreme_uo_stays)

    # Extreme Fluid
    extreme_fluid_stays = df[df["fluid_step"] > max_fluid]["stay_id"].unique()
    df = df[~df["stay_id"].isin(extreme_fluid_stays)]
    excluded_counts["extreme_fluid"] = len(extreme_fluid_stays)

    # Early Death
    first_times = df.groupby("stay_id")["timestamp"].min()
    last_times = df.groupby("stay_id")["timestamp"].max()
    morta = df.groupby("stay_id")["morta_hosp"].first()
    time_to_death = (last_times - first_times) / 3600
    early_deaths = morta[(morta == 1) & (time_to_death <= early_death_hours)].index
    df = df[~df["stay_id"].isin(early_deaths)]
    excluded_counts["early_death"] = len(early_deaths)

    print("\nExclusion Statistics:")
    print("-" * 50)
    for reason, count in excluded_counts.items():
        print(f"Excluded due to {reason}: {count}")
    print("-" * 50)

    return df


def add_infection_and_sepsis_flag(df):
    """
    Assign ``infection_active`` and ``sepsis`` columns using the Sepsis-3 definition.

    ``sepsis`` uses a 0/1/2 censoring scheme: 0 = no sepsis; 1 = onset timestep
    (first per stay where infection is active and SOFA \u2265 2); 2 = all subsequent
    timesteps after onset.

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory DataFrame. Must contain ``onset_time``, ``timestamp``,
        ``stay_id``, and ``sofa_score`` columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame sorted by ``(stay_id, timestamp)`` with
        ``infection_active`` (0/1) and ``sepsis`` (0/1/2) columns added.
        Temporary ``has_sepsis`` helper column is dropped before return.

    Notes
    -----
    Non-septic stays (``onset_time`` is NaN) always have ``sepsis = 0`` \u2014
    the ``infection_active`` check short-circuits to False for those rows.

    SOFA \u2265 2 alone is not sufficient; a confirmed infection onset is required.
    """
    print("Adding sepsis flags...")
    df = df.sort_values(["stay_id", "timestamp"])

    df["infection_active"] = 0

    is_actively_infected = df["onset_time"].notna() & (
        df["timestamp"] >= df["onset_time"]
    )
    df.loc[is_actively_infected, "infection_active"] = 1

    sepsis_condition = (df["infection_active"] == 1) & (df["sofa_score"] >= 2)

    df["sepsis"] = 0
    first_sepsis = df[sepsis_condition].groupby("stay_id").head(1).index
    df.loc[first_sepsis, "sepsis"] = 1

    # Cumsum propagates after the onset row: post-onset rows have has_sepsis == 1
    # and sepsis == 0, which is the exact mask for the censored label
    df["has_sepsis"] = df.groupby("stay_id")["sepsis"].cumsum()
    censored_mask = (df["has_sepsis"] == 1) & (df["sepsis"] == 0)
    df.loc[censored_mask, "sepsis"] = 2

    return df.drop(columns=["has_sepsis"])


def add_septic_shock_flag(df, shock_cfg):
    """
    Assign a ``septic_shock`` column using the Sepsis-3 definition, with the
    same 0/1/2 censoring scheme as ``sepsis``.

    Onset requires all five criteria to hold simultaneously:
    active sepsis (``sepsis`` ∈ {1, 2}), MAP below threshold, lactate above
    threshold, vasopressors active, and adequate rolling fluid resuscitation.

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory DataFrame. Must contain ``sepsis``, ``map``, ``lactate``,
        ``vaso_max``, ``fluid_step``, ``stay_id``, and ``timestamp`` columns.
        Requires ``sepsis`` to have been assigned by
        ``add_infection_and_sepsis_flag`` first.
    shock_cfg : dict
        Shock threshold configuration. Required keys:
        ``map_threshold_mmhg``, ``lactate_threshold_mmol``,
        ``fluid_resuscitation_ml``, ``fluid_window_timesteps``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame sorted by ``(stay_id, timestamp)`` with
        ``septic_shock`` (0/1/2) column added. Temporary ``has_shock`` and
        ``fluid_rolling_12h`` columns are dropped before return.

    Notes
    -----
    ``septic_shock = 2`` (post-onset censored) can appear without a preceding
    ``septic_shock = 1`` in the output if the onset timestep falls outside the
    observable trajectory window — matching the same censoring design as
    ``sepsis``.
    """
    map_threshold = shock_cfg["map_threshold_mmhg"]
    lactate_threshold = shock_cfg["lactate_threshold_mmol"]
    fluid_ml = shock_cfg["fluid_resuscitation_ml"]
    fluid_window = shock_cfg["fluid_window_timesteps"]

    print("Adding septic shock flags...")
    df = df.sort_values(["stay_id", "timestamp"])

    has_sepsis = df["sepsis"].isin([1, 2])

    # Rolling fluid sum over configured window, per stay
    df["fluid_rolling_12h"] = df.groupby("stay_id")["fluid_step"].transform(
        lambda x: x.rolling(fluid_window, min_periods=1).sum()
    )

    shock_clinical_criteria = (
        (df["map"] < map_threshold)
        & (df["lactate"] > lactate_threshold)
        & (df["vaso_max"] > 0)
        & (df["fluid_rolling_12h"] >= fluid_ml)
    )

    shock_condition = has_sepsis & shock_clinical_criteria

    df["septic_shock"] = 0
    first_shock = df[shock_condition].groupby("stay_id").head(1).index
    df.loc[first_shock, "septic_shock"] = 1

    df["has_shock"] = df.groupby("stay_id")["septic_shock"].cumsum()
    censored_mask = (df["has_shock"] == 1) & (df["septic_shock"] == 0)
    df.loc[censored_mask, "septic_shock"] = 2

    return df.drop(columns=["has_shock", "fluid_rolling_12h"])
