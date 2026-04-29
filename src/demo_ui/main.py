import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import pickle
import mlflow.xgboost
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config: Data & Model ───────────────────────────────────────────────────────
DATA_PATH   = "/home/joffray/repos/uni/dissertation/code/src/demo_ui/test_set_new2.parquet"
MODEL_PATH  = "/home/joffray/repos/uni/dissertation/code/mlruns/5/models/m-6f94a3404987472a88730dccbc5fd81d/artifacts/"

PATIENT_ID_COL     = "stay_id"
TIME_COL           = "timestep"
TARGET_COL         = "target"
N_DEMO_PATIENTS    = 5
PREDICTION_HORIZON = 1
PREDICTION_WINDOW  = 3

# ── Config: UI & Styling ───────────────────────────────────────────────────────
# Fonts
FONT_FAMILY_SANS = "'IBM Plex Sans', sans-serif"
FONT_FAMILY_MONO = "'IBM Plex Mono', monospace"

# CSS Settings
CSS_FONT_BASE       = "1rem"
CSS_FONT_METRIC_VAL = "1.8rem"
CSS_FONT_METRIC_LBL = "0.7rem"
CSS_FONT_PATIENT_HDR= "0.75rem"

# Plotly Font Sizes
PLOT_FONT_MAIN       = 30
PLOT_FONT_TICKS      = 30
PLOT_FONT_ANNOTATION = 30
PLOT_FONT_CALIB      = 11

# Colors
COLOR_BG_MAIN    = "#0d1117"
COLOR_BG_CARD    = "#161b22"
COLOR_TEXT_MAIN  = "#e6edf3"
COLOR_TEXT_MUTED = "#8b949e"
COLOR_GRID       = "#21262d"

COLOR_RISK_HIGH  = "#ff4d4f"
COLOR_RISK_MED   = "#faad14"
COLOR_RISK_LOW   = "#52c41a"
COLOR_INFO       = "#58a6ff"

FEATURE_COLORS   = ["#58a6ff", "#3fb950", "#d2a8ff", "#ffa657", "#79c0ff", "#56d364"]

IDS_TO_NAMES = {
    35134787: "Maria", 
    38613519: "James", 
    36349522: "Sarah", 
    38787960: "David", 
}

# ───────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Sepsis Risk Monitor", page_icon="🏥", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] {{ font-family: {FONT_FAMILY_SANS}; background-color: {COLOR_BG_MAIN}; color: {COLOR_TEXT_MAIN}; font-size: {CSS_FONT_BASE}; }}
.main {{ background-color: {COLOR_BG_MAIN}; }}
h1, h2, h3 {{ font-family: {FONT_FAMILY_MONO}; }}
.metric-card {{ background: {COLOR_BG_CARD}; border: 1px solid #30363d; border-radius: 8px; padding: 1.2rem 1.5rem; text-align: center; }}
.metric-label {{ font-family: {FONT_FAMILY_MONO}; font-size: {CSS_FONT_METRIC_LBL}; color: {COLOR_TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }}
.metric-value {{ font-family: {FONT_FAMILY_MONO}; font-size: {CSS_FONT_METRIC_VAL}; font-weight: 600; }}
.risk-HIGH   {{ color: {COLOR_RISK_HIGH}; }}
.risk-MEDIUM {{ color: {COLOR_RISK_MED}; }}
.risk-LOW    {{ color: {COLOR_RISK_LOW}; }}
.patient-header {{ font-family: {FONT_FAMILY_MONO}; font-size: {CSS_FONT_PATIENT_HDR}; color: {COLOR_TEXT_MUTED}; letter-spacing: 0.08em; margin-bottom: 0.25rem; }}
.info-box {{ background: #1c2333; border-left: 3px solid {COLOR_INFO}; border-radius: 4px; padding: 0.75rem 1rem; font-size: 0.8rem; color: {COLOR_TEXT_MUTED}; margin-bottom: 1rem; font-family: {FONT_FAMILY_MONO}; }}
</style>
""", unsafe_allow_html=True)

# ── Load data & model ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return mlflow.xgboost.load_model(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH).sort_values([PATIENT_ID_COL, TIME_COL])
    cohort = IDS_TO_NAMES.keys()
    return df[df[PATIENT_ID_COL].isin(cohort)].copy()

@st.cache_data
def get_predictions(_model, _df):
    f_cols = [c for c in _df.columns if c not in [PATIENT_ID_COL, TIME_COL, TARGET_COL, "sepsis"]]
    probs = _model.predict(xgb.DMatrix(_df[f_cols]))
    result = _df[[PATIENT_ID_COL, TIME_COL, TARGET_COL]].copy()
    result["pred_prob"] = probs
    return result, f_cols

@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

model = load_model()
df = load_data()
preds, feature_cols = get_predictions(model, df)
explainer = get_shap_explainer(model)

# ── Sidebar & Filtering ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Sepsis Risk Monitor")
    st.markdown("---")
    patient_ids = sorted(preds[PATIENT_ID_COL].unique())


# Use format_func to show the name in the UI
    selected_id = st.selectbox(
        "Select Patient", 
        options=patient_ids, 
        format_func=lambda x: IDS_TO_NAMES.get(x, f"Patient {x}")
    )    
    p_preds_full = preds[preds[PATIENT_ID_COL] == selected_id].sort_values(TIME_COL)
    p_data_full  = df[df[PATIENT_ID_COL] == selected_id].sort_values(TIME_COL)
    all_times = p_preds_full[TIME_COL].values

    st.markdown("---")
    st.markdown('<div class="patient-header">SIMULATION CONTROL</div>', unsafe_allow_html=True)
    current_t = st.select_slider("Current Simulation Timestep", options=all_times, value=all_times[0])
    selected_features = st.multiselect("Features to plot", options=feature_cols, default=None)

p_preds = p_preds_full[p_preds_full[TIME_COL] <= current_t]
p_data  = p_data_full[p_data_full[TIME_COL] <= current_t]

patient_age = p_data_full['age'].iloc[0] if 'age' in p_data_full.columns else "N/A"
patient_gender = p_data_full['gender'].iloc[0] if 'gender' in p_data_full.columns else "N/A"
gender_display = "Male" if str(patient_gender) in ['1', '1.0', 'M', 'Male'] else "Female" if str(patient_gender) in ['0', '0.0', 'F', 'Female'] else "Unknown"

# ── Metrics & Logic ───────────────────────────────────────────────────────────
peak_risk    = p_preds["pred_prob"].max()
final_risk   = p_preds["pred_prob"].iloc[-1]
true_outcome = int(p_preds_full[TARGET_COL].max())

# Timesteps where target==1: the window the model SHOULD predict high risk
# (sepsis is 1–3 steps ahead, so these are the pre-onset alarm rows)
onset_rows      = p_preds_full[p_preds_full[TARGET_COL] == 1]
if not onset_rows.empty:
    onset_window_t0 = onset_rows[TIME_COL].min()   # first t where target=1
    onset_window_t1 = min(onset_rows[TIME_COL].max(), onset_window_t0 + PREDICTION_WINDOW)   # last  t where target=1
else:
    onset_window_t0 = onset_window_t1 = None

# --- Calculate Clinical Sepsis Window (sepsis > 0) ---
sepsis_active_rows = p_data_full[p_data_full['sepsis'] > 0]
if not sepsis_active_rows.empty:
    sepsis_t0 = sepsis_active_rows[TIME_COL].min()
    sepsis_t1 = sepsis_active_rows[TIME_COL].max()
else:
    sepsis_t0 = sepsis_t1 = None

@st.cache_data
def get_calibration(_model):
    full_test = pd.read_parquet(DATA_PATH)
    f_cols = [c for c in full_test.columns if c not in [PATIENT_ID_COL, TIME_COL, TARGET_COL, "sepsis"]]
    full_probs = _model.predict(xgb.DMatrix(full_test[f_cols]))
    prob_true, prob_pred = calibration_curve(full_test[TARGET_COL], full_probs, n_bins=10)
    return prob_true, prob_pred

prob_true, prob_pred = get_calibration(model)
calibrated_frac = float(np.interp(final_risk, prob_pred, prob_true))

# Over full patient history (all timesteps revealed)
if p_preds_full[TARGET_COL].nunique() > 1:
    patient_auprc = average_precision_score(p_preds_full[TARGET_COL], p_preds_full["pred_prob"])
else:
    patient_auprc = None

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"# {IDS_TO_NAMES.get(selected_id, f'Patient {selected_id}')}")
st.markdown(
    f'<div class="patient-header">'
    f'ID: {selected_id} | {gender_display} | Age {int(patient_age) if patient_age != "N/A" else "N/A"}'
    f'</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)
for col, label, value, extra in [
    (c1, "CURRENT TIMESTEP", str(current_t), 'class="metric-value"'),
    (c2, "CURRENT RAW OUTPUT", f"{final_risk:.1%}", f'class="metric-value risk-{"HIGH" if final_risk >= 0.7 else "MEDIUM" if final_risk >= 0.4 else "LOW"}"'),
    (c3, "CALIBRATED FRACTION", f"{calibrated_frac:.1%}", f'class="metric-value risk-{"HIGH" if calibrated_frac >= 0.7 else "MEDIUM" if calibrated_frac >= 0.4 else "LOW"}"'),
    (c4, "PATIENT AUPRC", f"{patient_auprc:.2f}" if patient_auprc is not None else "N/A", 'class="metric-value"'),
]:
    col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div {extra}>{value}</div></div>', unsafe_allow_html=True)

# ── Main Risk Chart ────────────────────────────────────────────────────────────
n_features = len(selected_features)
fig = make_subplots(rows=1+n_features, cols=1, shared_xaxes=True, vertical_spacing=0.15, 
                    row_heights=[0.65] + [0.35/n_features]*n_features if n_features else [1.0])

full_x_range = [all_times.min(), all_times.max()]

fig.add_trace(go.Scatter(x=p_preds[TIME_COL], y=p_preds["pred_prob"], fill="tozeroy",
                         line=dict(color=COLOR_RISK_HIGH, width=2.5), name="Risk Prob"), row=1, col=1,)



if onset_window_t0 is not None:
    fig.add_vrect(
        x0=onset_window_t0, x1=onset_window_t1,
        fillcolor="rgba(250,173,20,0.2)",
        line=dict(color=COLOR_RISK_MED, width=2, dash="solid"),
        annotation_text="TARGET WINDOW",
        annotation_textangle=-90,
        annotation_position="inside left",
        annotation_font=dict(color=COLOR_RISK_MED, size=PLOT_FONT_ANNOTATION, family="IBM Plex Mono"),
        row=1, col=1,
    )

if sepsis_t0 is not None:
    fig.add_vrect(
        x0=sepsis_t0, x1=sepsis_t1,
        fillcolor="rgba(88, 166, 255, 0.15)", # Using COLOR_INFO with transparency
        line=dict(color=COLOR_INFO, width=2, dash="dot"),
        annotation_text="SEPSIS",
        annotation_textangle=-90,
        annotation_position="inside left",
        annotation_font=dict(color=COLOR_INFO, size=PLOT_FONT_ANNOTATION, family="IBM Plex Mono"),
        row=1, col=1,
    )


fig.update_yaxes(range=[0, 1], row=1, col=1, tickfont=dict(size=PLOT_FONT_TICKS), title_text="Predicted Risk", title_font=dict(size=PLOT_FONT_MAIN))

# ── Feature Subplots ───────────────────────────────────────────────────────────
for i, feat in enumerate(selected_features):
    color = FEATURE_COLORS[i % len(FEATURE_COLORS)] 
    
    fig.add_trace(go.Scatter(
        x=p_data[TIME_COL], 
        y=p_data[feat], 
        mode="lines", 
        line=dict(color=color, width=1.5), 
        name=feat,
        hovertemplate=f"<b>{feat}</b>: %{{y:.2f}}<extra></extra>"
    ), row=i+2, col=1)
    
    fig.update_yaxes(
        title_font=dict(size=PLOT_FONT_MAIN, color=color), 
        tickfont=dict(size=PLOT_FONT_TICKS),
        gridcolor=COLOR_GRID, 
        zeroline=False, 
        row=i+2, col=1
    )

fig.update_xaxes(
    range=full_x_range, gridcolor=COLOR_GRID, 
    title_text="Timestep", title_font=dict(size=PLOT_FONT_MAIN), 
    tickfont=dict(size=PLOT_FONT_TICKS), 
    row=n_features+1, col=1
)

fig.update_layout(
    height=400 + n_features*120, 
    paper_bgcolor=COLOR_BG_MAIN, plot_bgcolor=COLOR_BG_MAIN, 
    margin=dict(l=10, r=10, t=30, b=10), 
    font=dict(family="IBM Plex Mono", color=COLOR_TEXT_MUTED, size=PLOT_FONT_MAIN)
)
st.plotly_chart(fig, width='stretch')

# ── SHAP Reasoning ─────────────────────────────────────────────────────────────
st.markdown("---")
st.header("Current Risk Explanation")

obs = p_data_full[p_data_full[TIME_COL] == current_t][feature_cols]
if not obs.empty:
    shap_values = explainer(obs)
    sv = shap_values[0]
    
    full_df = pd.DataFrame({'name': sv.feature_names, 'val': sv.data, 'shaps': sv.values})
    full_df['abs'] = full_df['shaps'].abs()
    full_df = full_df.sort_values('abs', ascending=False)
    
    top_df = full_df.head(9).copy()
    other_impact = full_df.tail(len(full_df)-9)['shaps'].sum()
    
    plot_names = [f"{n} = {v:.2f}" for n, v in zip(top_df['name'], top_df['val'])] + ["Other features"]
    plot_shaps = list(top_df['shaps']) + [other_impact]
    
    plot_names.reverse()
    plot_shaps.reverse()

    fig_shap = go.Figure(go.Waterfall(
        orientation="h", measure=["relative"]*len(plot_names),
        y=plot_names, x=plot_shaps, base=sv.base_values,
        text=[f"{x:+.4f}" for x in plot_shaps], textposition="outside",
        decreasing={"marker": {"color": COLOR_INFO}}, 
        increasing={"marker": {"color": COLOR_RISK_HIGH}}
    ))
    fig_shap.update_layout(
        paper_bgcolor=COLOR_BG_MAIN, plot_bgcolor=COLOR_BG_MAIN, 
        font=dict(family="IBM Plex Mono", color=COLOR_TEXT_MUTED, size=PLOT_FONT_MAIN),
        margin=dict(l=10, r=80, t=10, b=10), height=400
    )
    st.plotly_chart(fig_shap, width='stretch')
else:
    st.warning("No data available for the selected timestep.")

# ── Calibration Curve ──────────────────────────────────────────────────────────
st.markdown("---")
st.header("Calibration Curve")

fig_cal = go.Figure()

fig_cal.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode="lines",
    line=dict(color="#30363d", width=1.5, dash="dash"),
    name="Perfectly Calibrated",
))

fig_cal.add_trace(go.Scatter(
    x=prob_pred, y=prob_true,
    mode="lines+markers",
    line=dict(color=COLOR_INFO, width=2),
    marker=dict(size=7),
    name="XGBoost",
    hovertemplate="Mean predicted: %{x:.2f}<br>Fraction positive: %{y:.2f}<extra></extra>",
))

fig_cal.add_vline(x=final_risk, line_dash="dot", line_color="rgba(255,77,79,0.7)", line_width=1.5)
fig_cal.add_hline(y=calibrated_frac, line_dash="dot", line_color="rgba(255,77,79,0.7)", line_width=1.5)

fig_cal.add_trace(go.Scatter(
    x=[final_risk], y=[calibrated_frac],
    mode="markers",
    marker=dict(color=COLOR_RISK_HIGH, size=12, symbol="circle", line=dict(color="#fff", width=2)),
    name=f"Current output ({final_risk:.1%} → {calibrated_frac:.1%})",
    hovertemplate=f"Model output: {final_risk:.1%}<br>Fraction positive: {calibrated_frac:.1%}<extra></extra>",
))

fig_cal.update_layout(
    paper_bgcolor=COLOR_BG_MAIN, plot_bgcolor=COLOR_BG_MAIN,
    font=dict(family="IBM Plex Mono", color=COLOR_TEXT_MUTED, size=PLOT_FONT_CALIB),
    xaxis=dict(title="Mean Predicted Probability", gridcolor=COLOR_GRID, range=[0, 1]),
    yaxis=dict(title="Fraction of Positives", gridcolor=COLOR_GRID, range=[0, 1]),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=10, r=10, t=10, b=10),
    height=350,
)

st.plotly_chart(fig_cal, width='stretch')