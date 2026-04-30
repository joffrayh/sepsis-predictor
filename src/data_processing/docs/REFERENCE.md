# Pipeline Reference

This document is the reference for the MIMIC-IV sepsis prediction pipeline. It covers two things:
1. **Column Dictionary** — every column that appears in intermediate or final pipeline outputs.
2. **Feature Explanations** — clinical context and pipeline logic for the more complex features.

---

## 1. Column Dictionary

### Final Output Columns

These columns appear in the final `sepsis_trajectories_4h.parquet`.

| Column | Full Name | Description | Units |
|---|---|---|---|
| `subject_id` | Patient identifier | Unique identifier for an individual patient across all hospital admissions | — |
| `hadm_id` | Hospital admission ID | Unique identifier for a specific hospital admission | — |
| `stay_id` | ICU stay ID | Unique identifier for a specific ICU stay within a hospital admission | — |
| `admittime` | Hospital admission time | Timestamp when the patient was admitted to the hospital | Unix timestamp / datetime |
| `dischtime` | Hospital discharge time | Timestamp when the patient was discharged from the hospital | Unix timestamp / datetime |
| `intime` | ICU admission time | Timestamp when the patient entered the ICU | Unix timestamp / datetime |
| `outtime` | ICU discharge time | Timestamp when the patient left the ICU | Unix timestamp / datetime |
| `onset_time` | Infection onset time | Timestamp determined by Sepsis-3 temporal criteria (antibiotics + culture) | Unix timestamp / datetime |
| `morta_90` | 90-day mortality | Binary flag: did the patient die within 90 days of admission? | 0 = No, 1 = Yes |
| `morta_hosp` | In-hospital mortality | Binary flag: did the patient die during the hospital stay? | 0 = No, 1 = Yes |
| `re_admission` | 30-day readmission flag | Binary indicator: was this admission within 30 days of previous discharge? | 0 = No, 1 = Yes |
| `charlson_comorbidity_index` | Charlson Comorbidity Index | Weighted comorbidity score estimating mortality risk | Integer score (0+) |
| `gender` | Gender | Patient gender | 0 = Male, 1 = Female |
| `abx_given` | Antibiotic given flag | Binary flag: were antibiotics active in this 4 h window? | 0 = No, 1 = Yes |
| `hours_since_first_abx` | Hours since first antibiotic | Hours elapsed since first antibiotic administration | Hours (null if none given) |
| `num_abx` | Number of antibiotics | Count of unique antibiotics active in this 4 h window | Integer count |
| `vaso_median` | Median vasopressor rate | Median norepinephrine-equivalent vasopressor dose in this window | μg/kg/min equivalent |
| `vaso_max` | Maximum vasopressor rate | Maximum norepinephrine-equivalent vasopressor dose in this window | μg/kg/min equivalent |
| `fluid_step` | Window fluid intake | IV fluid intake during this 4 h window (TEV-standardised) | mL |
| `fluid_total` | Cumulative fluid intake | Cumulative IV fluid intake up to this window (TEV-standardised) | mL |
| `uo_step` | Window urine output | Urine output during this 4 h window | mL |
| `uo_total` | Cumulative urine output | Cumulative urine output up to this window | mL |
| `balance` | Net fluid balance | `fluid_total − uo_total` | mL |
| `pf_ratio` | P/F ratio | PaO₂ / (FiO₂ / 100) — respiratory function index | mmHg |
| `shock_index` | Shock index | Heart rate / systolic BP | — |
| `sofa_resp` | SOFA respiratory sub-score | 0–4, derived from P/F ratio | Points |
| `sofa_coag` | SOFA coagulation sub-score | 0–4, derived from platelet count | Points |
| `sofa_liver` | SOFA liver sub-score | 0–4, derived from bilirubin | Points |
| `sofa_cv` | SOFA cardiovascular sub-score | 0–4, derived from MAP + vasopressor requirement | Points |
| `sofa_cns` | SOFA CNS sub-score | 0–4, derived from GCS | Points |
| `sofa_renal` | SOFA renal sub-score | 0–4, derived from creatinine / urine output | Points |
| `sofa_score` | Total SOFA score | Sum of all six SOFA sub-scores | Points (0–24) |
| `sirs_score` | SIRS criteria count | Count of SIRS criteria met (temp, HR, RR/PaCO₂, WBC) | Points (0–4) |
| `infection_active` | Infection active flag | Binary: is `onset_time` known and has `timestamp` reached or passed it? | 0 / 1 |
| `sepsis` | Sepsis flag | 0 = no sepsis, 1 = onset timestep (first row meeting infection_active=1 AND SOFA ≥ 2), 2 = post-onset censored | 0 / 1 / 2 |
| `septic_shock` | Septic shock flag | 0 = no shock, 1 = onset timestep (first row meeting active sepsis + MAP < 65 + lactate > 2), 2 = post-onset censored | 0 / 1 / 2 |
| `<lab>_measured` | Lab measured flag | Binary: was a real measurement recorded at this timestep (pre-imputation)? | 0 / 1 |
| `hours_since_<lab>` | Hours since last lab | Hours elapsed since the last real measurement (pre-imputation) | Hours (null if never) |

### Pipeline-Internal Columns

These columns are used during intermediate processing steps and do not appear in the final output.

| Column | Full Name | Description |
|---|---|---|
| `starttime` | Antibiotic start time | Timestamp when an antibiotic administration began |
| `charttime` | Event charted time | Timestamp when a clinical event (lab, culture, microbiology) was recorded |
| `chartdate` | Event charted date | Date (without time) of a microbiology event; used to fill missing `charttime` |
| `prev_dischtime` | Previous discharge time | Discharge time of the patient's previous stay; used for readmission calculation |
| `stay_count` | ICU stay count | Count of ICU stays per patient/admission; used for stay ID inference |
| `idx` | Row index | Temporary helper column preserving row positions during merge operations |
| `abx_idx` | Antibiotic event index | Temporary index tracking antibiotic rows during infection onset matching |
| `diff_hr` | Time difference | Absolute hours between antibiotic `starttime` and bacteriology `charttime` |

---

## 2. Feature Explanations

### Mechanical Ventilation

A binary flag indicating whether the patient is on mechanical ventilation at this timestep.

---

### Tidal Volume

Volume of air inhaled or exhaled in a single breath. Normal range: 500–1,000 mL.

---

### Minute Volume

Volume of air inhaled or exhaled per minute = tidal volume × respiratory rate. Normal range: 5–10 L/min.

---

### Oxygen Flow Rate

Rate of oxygen flow through the patient's respiratory system, in L/min.

---

### PEEP

Positive end-expiratory pressure — the airway pressure remaining at the end of expiration. Normal range: 5–15 cmH₂O.

---

### FiO₂

Fraction of inspired oxygen — critical for computing the P/F ratio used in SOFA scoring and assessing respiratory dysfunction.

When a direct FiO₂ measurement is missing, it is estimated from oxygen flow rate and delivery device using the following rules:

**Nasal cannula / no device**

| Flow (L/min) | FiO₂ |
|---|---|
| ≤ 1 | 24% |
| ≤ 2 | 28% |
| ≤ 3 | 32% |
| ≤ 4 | 36% |
| ≤ 5 | 40% |
| ≤ 6 | 44% |
| ≤ 8 | 50% |
| ≤ 10 | 55% |
| ≤ 12 | 62% |
| ≤ 15 | 70% |

**Face mask and similar devices**

| Flow (L/min) | FiO₂ |
|---|---|
| ≤ 4 | 36% |
| ≤ 6 | 40% |
| ≤ 8 | 58% |
| ≤ 10 | 66% |
| ≤ 12 | 69% |
| ≤ 15 | 75% |

**Non-rebreather mask**

| Flow (L/min) | FiO₂ |
|---|---|
| ≤ 6 | 60% |
| ≤ 8 | 70% |
| ≤ 10 | 80% |
| ≤ 15 | 90% |
| ≥ 15 | 100% |

**CPAP / BiPAP**

| Flow (L/min) | FiO₂ |
|---|---|
| < 10 | 60% |
| 10–15 | 80% |
| ≥ 15 | 100% |

**Oxymizer**

| Flow (L/min) | FiO₂ |
|---|---|
| < 5 | 40% |
| 5–10 | 60% |
| ≥ 10 | 80% |

Room air (no oxygen delivery): FiO₂ = **21%**.

---

### SOFA Score

Six-organ severity score used as the primary sepsis criterion (Sepsis-3). Each sub-score is 0–4 points:

| Sub-score | Derived from |
|---|---|
| Respiratory (`sofa_resp`) | P/F ratio (PaO₂ / FiO₂) |
| Cardiovascular (`sofa_cv`) | MAP + vasopressor requirement |
| Liver (`sofa_liver`) | Bilirubin |
| Neurological (`sofa_cns`) | GCS |
| Renal (`sofa_renal`) | Creatinine / urine output |
| Coagulation (`sofa_coag`) | Platelet count |

Total SOFA score = sum of sub-scores (0–24). Sepsis is defined as suspected infection + SOFA ≥ 2.

---

### Richmond Agitation-Sedation Scale (RASS)

A sedation/agitation scale scored −5 (unarousable) to +4 (combative). When GCS data is missing, GCS is imputed from the RASS score using a validated heuristic (see `utils/clinical_heuristics.py`).

---

### Charlson Comorbidity Index (CCI)

A weighted comorbidity score used to estimate 10-year mortality risk based on 17 conditions (e.g. diabetes, renal disease, malignancy). Missing values are filled with 0. Reference: [MDCalc CCI](https://www.mdcalc.com/calc/3917/charlson-comorbidity-index-cci).

---

### Vasopressors

All vasopressor doses are standardised to norepinephrine-equivalent units to allow cross-drug comparison. A rate of 0 means no vasopressors were administered in the window.

Vasopressors are required for the SOFA cardiovascular sub-score and the septic shock label.

---

### Fluid Standardisation (TEV)

Fluid volumes are converted to total equivalent volume (TEV) to account for differing tonicity:

| Type | Examples | TEV multiplier |
|---|---|---|
| Isotonic (1×) | NaCl 0.9%, Lactated Ringers, blood products, D5NS, D5LR | 1.0 |
| Hypotonic (0.5×) | NaCl 0.45%, D5 ½NS | 0.5 |
| Mannitol (2.75×) | Mannitol | 2.75 |
| NaCl 3% (3×) | Hypertonic saline 3% | 3.0 |
| Albumin 25% (5×) | Albumin 25% | 5.0 |
| NaHCO₃ 8.4% (6.66×) | Sodium bicarbonate 8.4% | 6.66 |
| NaCl 23.4% (8×) | Hypertonic saline 23.4% | 8.0 |

> **Note**: pre-admission fluid data is not available from MIMIC-IV, so `balance` values at early timesteps may underestimate true cumulative balance.

---

### Sepsis & Septic Shock Labels

Both `sepsis` and `septic_shock` use the same 0/1/2 censoring scheme:

| Value | Meaning |
|---|---|
| 0 | Criterion not yet met |
| 1 | **Onset timestep** — first timestep meeting the criteria |
| 2 | **Post-onset censored** — all subsequent timesteps for that stay |

**`sepsis` onset criteria** (all must hold at that timestep):
1. `infection_active = 1` (timestamp ≥ onset_time)
2. SOFA score ≥ 2

**`septic_shock` onset criteria** (all must hold at that timestep):
1. Active sepsis (`sepsis` = 1 or 2)
2. MAP < 65 mmHg
3. Lactate > 2 mmol/L
4. Vasopressors active (`vaso_max` > 0)
5. Rolling 12 h fluid intake ≥ 2,000 mL (3 × 4 h timesteps) — confirming hypotension persists despite adequate resuscitation
