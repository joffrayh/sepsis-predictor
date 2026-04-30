import numpy as np
import pandas as pd


def calculate_derived_variables(df):
    """
    Calculate derived variables such as P/F ratio (arterial oxygen pressure / FiO2),
    Shock Index (heart rate / systolic blood pressure), SOFA score and SIRS criteria.
    """
    df["gender"] = df["gender"] - 1
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

    # SOFA respiratory — vectorised pd.cut
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

    # SOFA cardiovascular — depends on both MAP and vaso, use np.select
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

    # SOFA renal — creatinine takes priority over UO
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

    # SIRS — vectorised
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


def apply_exclusion_criteria(df):
    """
    Apply exclusion criteria to filter out patients who do not meet the study
    criteria. Everything is vectorised. The exclusion criteria include:
    1. Extreme UO (>12000 ml in a 4h window)
    2. Extreme Fluid (>10000 ml in a 4h window)
    3. Early Death (death within 24 hours of first measurement)
    """
    print("Applying exclusion criteria...")
    excluded_counts = {}

    # 1. Extreme UO
    extreme_uo_stays = df[df["uo_step"] > 12000]["stay_id"].unique()
    df = df[~df["stay_id"].isin(extreme_uo_stays)]
    excluded_counts["extreme_uo"] = len(extreme_uo_stays)

    # 2. Extreme Fluid
    extreme_fluid_stays = df[df["fluid_step"] > 10000]["stay_id"].unique()
    df = df[~df["stay_id"].isin(extreme_fluid_stays)]
    excluded_counts["extreme_fluid"] = len(extreme_fluid_stays)

    # 3. Early Death (vectorised)
    first_times = df.groupby("stay_id")["timestamp"].min()
    last_times = df.groupby("stay_id")["timestamp"].max()
    morta = df.groupby("stay_id")["morta_hosp"].first()
    time_to_death = (last_times - first_times) / 3600
    early_deaths = morta[(morta == 1) & (time_to_death <= 24)].index
    df = df[~df["stay_id"].isin(early_deaths)]
    excluded_counts["early_death"] = len(early_deaths)

    print("\nExclusion Statistics:")
    print("-" * 50)
    for reason, count in excluded_counts.items():
        print(f"Excluded due to {reason}: {count}")
    print("-" * 50)

    return df


def add_septic_shock_flag(df):
    """
    Add septic shock flags based on the Sepsis-3 criteria, which defines septic
    shock as sepsis with persistent hypotension requiring vasopressors to
    maintain MAP ≥ 65 mm Hg, and having a serum lactate level > 2 mmol/L despite
    adequate volume resuscitation.
    We will use the following logic to determine septic shock onset and censoring:
    1. Patient has infection (onset_time is not NaN)
    2. Calculate rolling fluids over a 12-hour window (3 time steps if using 4h timesteps).
    3. Identify time steps where rolling fluids ≥ 2000 ml, MAP < 65 mm Hg,
         and lactic acid > 2 mmol/L.
    4. Mark the first time step meeting these criteria as septic shock onset (flag = 1).
    5. Mark all subsequent time steps for that patient as septic shock (flag = 2)
        to indicate censoring after onset.
    """
    print("Adding septic shock flags...")
    df = df.sort_values(["stay_id", "timestamp"])

    has_sepsis = df["sepsis"].isin([1, 2])

    shock_clinical_criteria = (df["lactate"] > 2.0) & (df["map"] < 65)

    shock_condition = has_sepsis & shock_clinical_criteria

    df["septic_shock"] = 0
    first_shock = df[shock_condition].groupby("stay_id").head(1).index
    df.loc[first_shock, "septic_shock"] = 1

    df["has_shock"] = df.groupby("stay_id")["septic_shock"].cumsum()
    censored_mask = (df["has_shock"] == 1) & (df["septic_shock"] == 0)
    df.loc[censored_mask, "septic_shock"] = 2

    return df.drop(columns=["has_shock"])


def add_infection_and_sepsis_flag(df):
    """
    Add sepsis flags based on Sepsis-3 criteria: SOFA >= 2 AND confirmed
    infection (onset_time is not NaN). Non-septic patients (onset_time is NaN)
    will always have sepsis = 0.
    Flags: 0 = no sepsis, 1 = onset timestep, 2 = post-onset (censored).
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

    df["has_sepsis"] = df.groupby("stay_id")["sepsis"].cumsum()
    censored_mask = (df["has_sepsis"] == 1) & (df["sepsis"] == 0)
    df.loc[censored_mask, "sepsis"] = 2

    return df.drop(columns=["has_sepsis"])
