import shap
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "/home/joffray/repos/uni/dissertation/code/mlruns/5/0be7d597c6084dac8e93d27cdde56c1a/artifacts/best_xgboost.pkl"
DATA_PATH  = "/home/joffray/repos/uni/dissertation/code/src/demo_ui/test_set.parquet"

PATIENT_ID_COL     = "stay_id"
TIME_COL           = "timestep"
TARGET_COL         = "target"
N_DEMO_PATIENTS    = 5
PREDICTION_HORIZON = 3 

FEATURE_COLS = ["heart_rate", "respiratory_rate", "temp_C"]
# ───────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Sepsis Risk Monitor", page_icon="🏥", layout="wide")

# (CSS Styling remains the same as your original)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background-color: #0d1117; color: #e6edf3; }
.main { background-color: #0d1117; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
.metric-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 1.2rem 1.5rem; text-align: center; }
.metric-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 600; }
.risk-HIGH   { color: #ff4d4f; }
.risk-MEDIUM { color: #faad14; }
.risk-LOW    { color: #52c41a; }
.patient-header { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #8b949e; letter-spacing: 0.08em; margin-bottom: 0.25rem; }
.info-box { background: #1c2333; border-left: 3px solid #388bfd; border-radius: 4px; padding: 0.75rem 1rem; font-size: 0.8rem; color: #8b949e; margin-bottom: 1rem; font-family: 'IBM Plex Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ── Load data & model ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH).sort_values([PATIENT_ID_COL, TIME_COL])
    pos_ids = df[df[TARGET_COL] == 1][PATIENT_ID_COL].unique()
    neg_ids = df[df[TARGET_COL] == 0][PATIENT_ID_COL].unique()
    rng = np.random.default_rng(42)
    n_pos = min(3, len(pos_ids))
    cohort = np.concatenate([rng.choice(pos_ids, size=n_pos, replace=False), rng.choice(neg_ids, size=N_DEMO_PATIENTS-n_pos, replace=False)])
    return df[df[PATIENT_ID_COL].isin(cohort)].copy()

@st.cache_data
def get_predictions(_model, _df):
    f_cols = [c for c in _df.columns if c not in [PATIENT_ID_COL, TIME_COL, TARGET_COL, "sepsis"]]
    probs = _model.predict_proba(_df[f_cols])[:, 1]
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
    selected_id = st.selectbox("Patient", options=patient_ids, format_func=lambda x: f"Patient {x}")
    
    # Pre-filter patient data for the slider range
    p_preds_full = preds[preds[PATIENT_ID_COL] == selected_id].sort_values(TIME_COL)
    p_data_full  = df[df[PATIENT_ID_COL] == selected_id].sort_values(TIME_COL)
    all_times = p_preds_full[TIME_COL].values

    st.markdown("---")
    st.markdown('<div class="patient-header">SIMULATION CONTROL</div>', unsafe_allow_html=True)
    current_t = st.select_slider("Current Simulation Hour", options=all_times, value=all_times[0])
    
    threshold = st.slider("Alert threshold", 0.1, 0.9, 0.5, 0.05)
    selected_features = st.multiselect("Features to plot", options=FEATURE_COLS, default=FEATURE_COLS[:3])

# Filter data to the "Current Simulation Time"
p_preds = p_preds_full[p_preds_full[TIME_COL] <= current_t]
p_data  = p_data_full[p_data_full[TIME_COL] <= current_t]

# Extract demographics from the first available row for this patient
# Note: Ensure "age" and "gender" (or similar) exist in your parquet file columns
patient_age = p_data_full['age'].iloc[0] if 'age' in p_data_full.columns else "N/A"
patient_gender = p_data_full['gender'].iloc[0] if 'gender' in p_data_full.columns else "N/A"

# Format gender for display (handling 0/1 or strings)
gender_display = "Male" if str(patient_gender) in ['1', '1.0', 'M', 'Male'] else "Female" if str(patient_gender) in ['0', '0.0', 'F', 'Female'] else "Unknown"

# ── Metrics & Logic ───────────────────────────────────────────────────────────
peak_risk    = p_preds["pred_prob"].max()
final_risk   = p_preds["pred_prob"].iloc[-1]
true_outcome = int(p_preds_full[TARGET_COL].max()) # Keep overall outcome for card
n_alerts     = int((p_preds["pred_prob"] >= threshold).sum())

# Sepsis Markers
onset_rows      = p_preds_full[p_preds_full[TARGET_COL] == 1]
first_label_t   = onset_rows[TIME_COL].iloc[0] if not onset_rows.empty else None
onset_window_t0 = first_label_t + 1 if first_label_t is not None else None
onset_window_t1 = first_label_t + PREDICTION_HORIZON if first_label_t is not None else None

# Alert Markers
alert_rows    = p_preds[p_preds["pred_prob"] >= threshold]
first_alert_t = alert_rows[TIME_COL].iloc[0] if not alert_rows.empty else None

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"# Patient {selected_id}")
# st.markdown(f'<div class="patient-header">SIMULATED VIEW AT HOUR {current_t}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="patient-header">'
    f'{gender_display} | Age {int(patient_age)} | {len(p_preds_full)} Timtesteps Total'
    f'</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, = st.columns(4)
# (Metric card logic is the same as your original, using peak_risk and final_risk)
for col, label, value, extra in [
    (c1, "CURRENT TIMESTEP", str(current_t), 'class="metric-value"'),
    (c2, "CURRENT RISK", f"{final_risk:.1%}", f'class="metric-value risk-{"HIGH" if final_risk >= 0.7 else "MEDIUM" if final_risk >= 0.4 else "LOW"}"'),
    (c3, "TRUE OUTCOME", "SEPSIS" if true_outcome else "NO SEPSIS", 'class="metric-value risk-HIGH"' if true_outcome else 'class="metric-value risk-LOW"'),
    (c4, "ALERTS FIRED", str(n_alerts), 'class="metric-value"'),
]:
    col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div {extra}>{value}</div></div>', unsafe_allow_html=True)

# ── Main Risk Chart ────────────────────────────────────────────────────────────
n_features = len(selected_features)
fig = make_subplots(rows=1+n_features, cols=1, shared_xaxes=True, vertical_spacing=0.04, 
                    row_heights=[0.55] + [0.45/n_features]*n_features if n_features else [1.0])

# Fix X-axis range so it doesn't jump
full_x_range = [all_times.min(), all_times.max()]

# Risk Curve
fig.add_trace(go.Scatter(x=p_preds[TIME_COL], y=p_preds["pred_prob"], fill="tozeroy",
                         line=dict(color="#ff4d4f", width=2.5), name="Risk Prob"), row=1, col=1)

# Alert markers (visible only if alerts happened before current_t)
alert_mask = p_preds["pred_prob"] >= threshold
if alert_mask.any():
    fig.add_trace(go.Scatter(x=p_preds[TIME_COL][alert_mask], y=p_preds["pred_prob"][alert_mask],
                             mode="markers", marker=dict(color="#ff4d4f", size=6), name="Alert"), row=1, col=1)

# Annotations (Vertical lines only appear if we've passed that timepoint)
if first_alert_t is not None:
    fig.add_vline(x=first_alert_t, line_dash="dash", line_color="#ff4d4f", row=1, col=1)

if first_label_t is not None and current_t >= first_label_t:
    fig.add_vline(x=first_label_t, line_color="#faad14", row=1, col=1)
# ── Feature Subplots with Unique Colors ────────────────────────────────────────
# Use the color palette defined in your config
FEATURE_COLORS = ["#58a6ff", "#3fb950", "#d2a8ff", "#ffa657", "#79c0ff", "#56d364"]

for i, feat in enumerate(selected_features):
    # The modulo (%) ensures we don't index out of bounds if you add many features
    color = FEATURE_COLORS[i % len(FEATURE_COLORS)] 
    
    fig.add_trace(go.Scatter(
        x=p_data[TIME_COL], 
        y=p_data[feat], 
        mode="lines", 
        line=dict(color=color, width=1.5), 
        name=feat,
        hovertemplate=f"<b>{feat}</b>: %{{y:.2f}}<extra></extra>"
    ), row=i+2, col=1) # Starts at row 2 because row 1 is the Risk Curve

    # Mirror the onset window on feature subplots for visual alignment
    if onset_window_t0 is not None:
        fig.add_vrect(
            x0=onset_window_t0, x1=onset_window_t1,
            fillcolor="rgba(250,173,20,0.05)",
            line=dict(color="rgba(250,173,20,0.2)", width=1, dash="dot"),
            row=i+2, col=1,
        )
    
    fig.update_yaxes(
        title_text=feat, 
        title_font=dict(size=10, color=color), # Match axis title to line color
        gridcolor="#21262d", 
        zeroline=False, 
        row=i+2, col=1
    )
fig.update_xaxes(range=full_x_range, gridcolor="#21262d")
fig.update_layout(height=400 + n_features*120, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", 
                  margin=dict(l=10, r=10, t=30, b=10), font=dict(family="IBM Plex Mono", color="#8b949e"))
st.plotly_chart(fig, use_container_width=True)

# ── SHAP Reasoning ─────────────────────────────────────────────────────────────
st.markdown("---")
st.header("Current Risk Explanation")

# Explain the CURRENT observation
obs = p_data_full[p_data_full[TIME_COL] == current_t][feature_cols]
if not obs.empty:
    shap_values = explainer(obs)
    sv = shap_values[0]
    
    # Calculate top 9 + "Other"
    full_df = pd.DataFrame({'name': sv.feature_names, 'val': sv.data, 'shaps': sv.values})
    full_df['abs'] = full_df['shaps'].abs()
    full_df = full_df.sort_values('abs', ascending=False)
    
    top_df = full_df.head(9).copy()
    other_impact = full_df.tail(len(full_df)-9)['shaps'].sum()
    
    plot_names = [f"{n} = {v:.2f}" for n, v in zip(top_df['name'], top_df['val'])] + ["Other features"]
    plot_shaps = list(top_df['shaps']) + [other_impact]
    
    # Reverse for better visual flow (top impact at top)
    plot_names.reverse(); plot_shaps.reverse()

    fig_shap = go.Figure(go.Waterfall(
        orientation="h", measure=["relative"]*len(plot_names),
        y=plot_names, x=plot_shaps, base=sv.base_values,
        text=[f"{x:+.4f}" for x in plot_shaps], textposition="outside",
        decreasing={"marker": {"color": "#58a6ff"}}, 
        increasing={"marker": {"color": "#ff4d4f"}}
    ))
    fig_shap.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", 
                          font=dict(family="IBM Plex Mono", color="#8b949e"),
                          margin=dict(l=10, r=80, t=10, b=10), height=400)
    st.plotly_chart(fig_shap, use_container_width=True)
else:
    st.warning("No data available for the selected timestep.")