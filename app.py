"""
WildGuard AI — Wildlife-Vehicle Collision Risk Prediction System
Main Streamlit Application with Real-Time Data Pipeline
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title   = "WildGuard AI — Wildlife Risk Prediction",
    page_icon    = "🐾",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS — Palantir Foundry Aesthetic ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
  --bg:         #060a10;
  --bg-alt:     #0a0f18;
  --surface:    #0d1320;
  --surface-2:  #111827;
  --border:     #1a2332;
  --border-hi:  #243044;
  --cyan:       #00e5ff;
  --cyan-dim:   #0091a3;
  --cyan-glow:  rgba(0,229,255,0.12);
  --amber:      #ffb020;
  --amber-dim:  #a67400;
  --red:        #ff3d5a;
  --red-dim:    #8b1a2b;
  --green:      #00e676;
  --green-dim:  #007a3d;
  --blue:       #4da6ff;
  --purple:     #9d7aff;
  --orange:     #ff7043;
  --low:        #00e676;
  --med:        #ffb020;
  --high:       #ff3d5a;
  --crit:       #d500f9;
  --text:       #c8d6e5;
  --text-hi:    #e8f0fa;
  --muted:      #5a6d82;
  --muted-lo:   #3a4a5c;
}

/* ─── Base ─────────────────────────────────────────────────────────────── */
html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Inter', -apple-system, sans-serif;
}

/* Subtle scanline overlay */
.stApp::before {
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background:
    repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,229,255,0.008) 2px, rgba(0,229,255,0.008) 4px);
  pointer-events: none;
  z-index: 0;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.2rem 1.8rem !important; max-width: 100% !important; }

/* ─── Keyframes ─────────────────────────────────────────────────────── */
@keyframes pulse-glow {
  0%, 100% { opacity: 0.6; }
  50% { opacity: 1; }
}
@keyframes scan-line {
  0% { top: -2px; }
  100% { top: 100%; }
}
@keyframes data-fade-in {
  0% { opacity: 0; transform: translateY(6px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes border-pulse {
  0%, 100% { border-color: var(--border); }
  50% { border-color: var(--border-hi); }
}

/* ─── HUD Metric Cards ─────────────────────────────────────────────── */
.metric-card {
  position: relative;
  background: linear-gradient(145deg, var(--surface), var(--bg-alt));
  border: 1px solid var(--border);
  padding: 1rem 1.2rem;
  margin: 0.25rem 0;
  animation: data-fade-in 0.4s ease-out;
  overflow: hidden;
}
/* Corner brackets — HUD style */
.metric-card::before, .metric-card::after {
  content: '';
  position: absolute;
  width: 12px;
  height: 12px;
  border-color: var(--cyan);
  border-style: solid;
}
.metric-card::before {
  top: 0; left: 0;
  border-width: 1px 0 0 1px;
}
.metric-card::after {
  bottom: 0; right: 0;
  border-width: 0 1px 1px 0;
}
.metric-card h3 {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.6rem;
  color: var(--muted);
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin: 0 0 0.5rem 0;
  font-weight: 500;
}
.metric-card .value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.8rem;
  font-weight: 600;
  color: var(--cyan);
  line-height: 1;
  text-shadow: 0 0 20px rgba(0,229,255,0.2);
}
.metric-card .sub {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.65rem;
  color: var(--muted);
  margin-top: 0.4rem;
  letter-spacing: 0.05em;
}

/* ─── Risk Badge ───────────────────────────────────────────────────── */
.risk-badge {
  display: inline-block;
  padding: 0.25rem 0.8rem;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
  font-size: 0.85rem;
  letter-spacing: 0.08em;
  border: 1px solid;
  text-transform: uppercase;
}

/* ─── Section Headers — Classified Document Style ─────────────────── */
.section-header {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.62rem;
  color: var(--cyan-dim);
  letter-spacing: 0.22em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.4rem;
  margin: 1.5rem 0 0.7rem 0;
  position: relative;
}
.section-header::before {
  content: '▸';
  margin-right: 0.5rem;
  color: var(--cyan);
}

/* ─── Source Cards ──────────────────────────────────────────────────── */
.source-card {
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 0.9rem 1.1rem;
  margin: 0.4rem 0;
  position: relative;
}
.source-card::before {
  content: '';
  position: absolute;
  left: 0; top: 0; bottom: 0;
  width: 2px;
  background: var(--cyan);
}
.source-card .source-name {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text-hi);
  margin-bottom: 0.2rem;
}
.source-card .source-url {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.62rem;
  color: var(--muted);
  word-break: break-all;
  margin-bottom: 0.3rem;
}
.source-card .source-features {
  font-size: 0.68rem;
  color: var(--cyan);
  font-family: 'IBM Plex Mono', monospace;
}
.status-success { color: var(--green); }
.status-fallback { color: var(--amber); }
.status-error { color: var(--red); }

/* ─── Sidebar — Dark Ops Panel ─────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #060a10 0%, #0a0f18 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"]::after {
  content: '';
  position: absolute;
  right: 0; top: 0; bottom: 0;
  width: 1px;
  background: linear-gradient(180deg, transparent, var(--cyan-dim), transparent);
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p {
  color: var(--text) !important;
  font-size: 0.78rem !important;
  font-family: 'IBM Plex Mono', monospace !important;
}

/* Navigation radio items */
[data-testid="stSidebar"] .stRadio label {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.04em !important;
  color: var(--text) !important;
  padding: 0.35rem 0 !important;
  transition: all 0.15s ease !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
  color: var(--cyan) !important;
}
[data-testid="stSidebar"] .stRadio [data-checked="true"] + label,
[data-testid="stSidebar"] input[type="radio"]:checked + label {
  color: var(--cyan) !important;
}

/* ─── Form Elements ────────────────────────────────────────────────── */
.stSelectbox > div > div {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.8rem !important;
}
.stSlider [data-baseweb="slider"] { background: var(--border) !important; }

.stTabs [data-baseweb="tab-list"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 0;
}
.stTabs [data-baseweb="tab"] {
  color: var(--muted) !important;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.stTabs [aria-selected="true"] {
  color: var(--cyan) !important;
  border-bottom: 2px solid var(--cyan) !important;
}

/* ─── Buttons — Technical Style ────────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, rgba(0,229,255,0.15), rgba(0,229,255,0.05)) !important;
  color: var(--cyan) !important;
  border: 1px solid var(--cyan-dim) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-weight: 600 !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  border-radius: 0 !important;
  padding: 0.6rem 2rem !important;
  transition: all 0.2s ease !important;
  position: relative !important;
  overflow: hidden !important;
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(0,229,255,0.25), rgba(0,229,255,0.1)) !important;
  box-shadow: 0 0 20px rgba(0,229,255,0.15) !important;
  border-color: var(--cyan) !important;
  transform: none !important;
}
.stButton > button:active {
  background: rgba(0,229,255,0.3) !important;
}

/* Metric value override */
div[data-testid="stMetricValue"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 1.6rem !important;
  color: var(--cyan) !important;
}

/* ─── Markdown text overrides ──────────────────────────────────────── */
.stMarkdown h1 { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; }
.stMarkdown h2 { font-family: 'Inter', sans-serif !important; font-weight: 600 !important; }
.stMarkdown h3 { font-family: 'IBM Plex Mono', monospace !important; font-weight: 500 !important; }

/* Dataframe / table styling */
.stDataFrame { border: 1px solid var(--border) !important; }
.stDataFrame th {
  background: var(--surface) !important;
  color: var(--cyan) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.7rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
}

/* ─── Divider — Subtle ─────────────────────────────────────────────── */
hr { border-color: var(--border) !important; opacity: 0.5 !important; }

/* ─── Live Indicator ───────────────────────────────────────────────── */
.live-dot {
  display: inline-block;
  width: 6px;
  height: 6px;
  background: var(--green);
  border-radius: 50%;
  animation: pulse-glow 2s ease-in-out infinite;
  margin-right: 6px;
  box-shadow: 0 0 6px var(--green);
}

/* ─── Scrollbar ────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted-lo); }

/* ─── Expander ─────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.72rem !important;
  color: var(--text) !important;
  letter-spacing: 0.06em !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load model artifacts ──────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR  = BASE_DIR / "data"

@st.cache_resource
def load_model():
    from models.train import WildlifeRiskModel, train_and_save
    m = WildlifeRiskModel()
    if not (MODEL_DIR / "xgb_model.pkl").exists():
        train_and_save()
    m.load()
    return m

@st.cache_data
def load_dataset():
    if (MODEL_DIR / "dataset.parquet").exists():
        return pd.read_parquet(MODEL_DIR / "dataset.parquet")
    else:
        from data.generate_data import generate_dataset
        df = generate_dataset(12_000)
        df.to_parquet(MODEL_DIR / "dataset.parquet", index=False)
        return df

model = load_model()
df    = load_dataset()

# Parse multi-model metrics
if "xgb" in model.metrics and isinstance(model.metrics["xgb"], dict):
    xgb_metrics = model.metrics["xgb"]
    rf_metrics  = model.metrics.get("rf", {})
    shap_imp    = model.metrics.get("shap_importance", [])
    rf_imp      = model.metrics.get("rf_importance", [])
    best_model  = model.metrics.get("best_model", "XGBoost")
else:
    # Backward compat with old single-model format
    xgb_metrics = model.metrics
    rf_metrics  = {}
    shap_imp    = model.metrics.get("shap_importance", [])
    rf_imp      = []
    best_model  = "XGBoost"

# ── Sidebar — Command Console ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.2rem 0 1rem 0;'>
      <div style='font-size: 0.55rem; color: #3a4a5c; letter-spacing: 0.25em; font-family: "IBM Plex Mono", monospace; margin-bottom: 0.6rem;'>SYSTEM ONLINE</div>
      <div style='position:relative; display:inline-block;'>
        <div style='font-size:2.2rem; filter: drop-shadow(0 0 8px rgba(0,229,255,0.3));'>🐾</div>
      </div>
      <div style='font-family:"JetBrains Mono",monospace; font-size:1rem; color:#00e5ff; font-weight:600; margin-top:0.4rem; letter-spacing:0.05em; text-shadow: 0 0 15px rgba(0,229,255,0.3);'>WILDGUARD</div>
      <div style='font-family:"IBM Plex Mono",monospace; font-size:0.58rem; color:#5a6d82; letter-spacing:0.2em; margin-top:0.15rem;'>RISK INTELLIGENCE v3.0</div>
      <div style='margin-top:0.5rem; padding:0.25rem 0.8rem; display:inline-block; border: 1px solid rgba(0,230,118,0.3); background:rgba(0,230,118,0.05);'>
        <span class='live-dot'></span>
        <span style='font-family:"IBM Plex Mono",monospace; font-size:0.55rem; color:#00e676; letter-spacing:0.15em;'>PIPELINE ACTIVE</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 Live Risk Predictor", "🗺 Risk Map",
         "📊 Analytics", "⚡ Future Hotspots", "🧠 Model Insights", "📡 Data Sources"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    xgb_auc = xgb_metrics.get('roc_auc', 0)
    rf_auc  = rf_metrics.get('roc_auc', 0)
    st.markdown(f"""
    <div style='padding:0.5rem 0;'>
      <div style='font-family:"IBM Plex Mono",monospace; font-size:0.52rem; color:#3a4a5c; letter-spacing:0.2em; margin-bottom:0.5rem;'>▸ MODEL STATUS</div>
      <div class='metric-card' style='margin-bottom:0.2rem;'>
        <h3>XGBoost Classifier</h3>
        <div class='value' style='font-size:0.85rem; color:#00e5ff;'>AUC {xgb_auc:.4f}</div>
        <div class='sub'>gradient boosted trees</div>
      </div>
      <div class='metric-card' style='margin-bottom:0.2rem;'>
        <h3>Random Forest</h3>
        <div class='value' style='font-size:0.85rem; color:#4da6ff;'>AUC {rf_auc:.4f}</div>
        <div class='sub'>ensemble bagging</div>
      </div>
      <div class='metric-card'>
        <h3>Active Model</h3>
        <div class='value' style='font-size:0.78rem; color:#00e676;'>◆ {best_model}</div>
        <div class='sub'>highest validation AUC</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    from utils.helpers import (
        hourly_risk_chart, species_risk_chart,
        season_road_heatmap, rolling_trend_chart, ndvi_risk_scatter,
        confusion_matrix_chart, model_comparison_chart, roc_comparison_chart,
        PALETTE
    )

    st.markdown(f"""
    <div style='margin-bottom:1.5rem;'>
      <div style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; color:#3a4a5c; letter-spacing:0.25em; margin-bottom:0.3rem;'>
        ▸ CLASSIFIED // RISK INTELLIGENCE OVERVIEW // {datetime.now().strftime('%Y-%m-%d %H:%M UTC+5:30')}
      </div>
      <h1 style='font-family:"Inter",sans-serif; font-size:1.8rem; font-weight:600; margin:0 0 0.15rem 0; color:#e8f0fa;'>
        Wildlife-Vehicle Collision
        <span style='color:#00e5ff;'>Risk Intelligence</span>
      </h1>
      <p style='color:#5a6d82; font-size:0.78rem; margin:0; font-family:"IBM Plex Mono",monospace; letter-spacing:0.03em;'>
        Dual-model ensemble (XGBoost + Random Forest) · {len(df):,} training records · 7-source real-time pipeline
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ────────────────────────────────────────────────────────────────
    acc_rate  = df["accident"].mean()
    high_risk = (df["risk_score"] > 0.65).sum()
    top_sp    = df[df["accident"]==1]["species"].value_counts().idxmax()
    top_road  = df[df["accident"]==1]["road_type"].value_counts().idxmax()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (c1, "Total Records",    f"{len(df):,}",                     "training dataset"),
        (c2, "Accident Rate",    f"{acc_rate:.1%}",                  "binary target"),
        (c3, "High-Risk Zones",  f"{high_risk:,}",                   "score > 0.65"),
        (c4, "Riskiest Species", top_sp.capitalize(),                 "highest frequency"),
        (c5, "XGB AUC",          f"{xgb_metrics.get('roc_auc',0):.4f}", "gradient boosting"),
        (c6, "RF AUC",           f"{rf_metrics.get('roc_auc',0):.4f}",  "random forest"),
    ]
    for col, title, val, sub in kpis:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <h3>{title}</h3>
              <div class='value' style='font-size:1.6rem;'>{val}</div>
              <div class='sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model comparison row ──────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📈 Model Performance Comparison — XGBoost vs Random Forest</div>", unsafe_allow_html=True)
    col_mc, col_roc = st.columns(2)
    with col_mc:
        if rf_metrics:
            st.plotly_chart(model_comparison_chart(xgb_metrics, rf_metrics), use_container_width=True)
    with col_roc:
        if rf_metrics:
            st.plotly_chart(roc_comparison_chart(xgb_metrics, rf_metrics), use_container_width=True)

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        st.plotly_chart(hourly_risk_chart(df), use_container_width=True)
    with col_b:
        st.plotly_chart(species_risk_chart(df), use_container_width=True)

    # ── Charts Row 2 ──────────────────────────────────────────────────────────
    col_c, col_d = st.columns([1, 1.2])
    with col_c:
        st.plotly_chart(season_road_heatmap(df), use_container_width=True)
    with col_d:
        st.plotly_chart(ndvi_risk_scatter(df), use_container_width=True)

    # ── Rolling trend ─────────────────────────────────────────────────────────
    st.plotly_chart(rolling_trend_chart(df), use_container_width=True)

    # ── Confusion matrices side by side ───────────────────────────────────────
    st.markdown("<div class='section-header'>Confusion Matrices</div>", unsafe_allow_html=True)
    col_e, col_f = st.columns(2)
    with col_e:
        cm_xgb = xgb_metrics.get("confusion", [[0,0],[0,0]])
        st.plotly_chart(confusion_matrix_chart(cm_xgb, "XGBoost — Confusion Matrix"), use_container_width=True)
    with col_f:
        if rf_metrics:
            cm_rf = rf_metrics.get("confusion", [[0,0],[0,0]])
            st.plotly_chart(confusion_matrix_chart(cm_rf, "Random Forest — Confusion Matrix"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE RISK PREDICTOR (Real-Time Data Pipeline)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Risk Predictor":
    from utils.helpers import (
        dual_risk_gauge, shap_waterfall, get_risk_level,
        data_source_status_chart, PALETTE
    )
    from data.realtime_extractor import RealtimeDataExtractor, geocode_place, PROTECTED_AREAS, WILDLIFE_CORRIDORS

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 0.3rem 0;'>
      🔮 Live <span style='color:#FF6B35;'>Risk Prediction</span> Pipeline
    </h1>
    <p style='color:#8B949E; font-size:0.82rem; margin:0 0 1.2rem 0;'>
      Search any place in India → extracts real-time data from APIs, news sites & govt portals → dual-model prediction
    </p>
    """, unsafe_allow_html=True)

    # ── Location inputs: Search + Presets + Manual ────────────────────────────
    st.markdown("<div class='section-header'>📍 Search Location</div>", unsafe_allow_html=True)

    # Search bar
    search_col, search_btn_col = st.columns([4, 1])
    with search_col:
        place_query = st.text_input(
            "🔍 Search any place in India",
            placeholder="e.g., Bandipur Tiger Reserve, Mysore-Ooty Highway, Wayanad...",
            key="place_search"
        )
    with search_btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("🔎 Search", use_container_width=True)

    # Handle geocoding
    resolved_lat, resolved_lon = None, None
    resolved_name = ""
    if search_clicked and place_query:
        with st.spinner(f"🌍 Geocoding '{place_query}' via Nominatim (OpenStreetMap)..."):
            geo_result = geocode_place(place_query)
        if geo_result and geo_result.get("found"):
            resolved_lat = geo_result["lat"]
            resolved_lon = geo_result["lon"]
            resolved_name = geo_result.get("display_name", "")
            st.success(f"📍 **Found:** {resolved_name}")
            st.markdown(f"<div style='font-size:0.78rem; color:var(--muted);'>Lat: {resolved_lat:.5f} | Lon: {resolved_lon:.5f} | Type: {geo_result.get('place_type','')}</div>", unsafe_allow_html=True)
            if len(geo_result.get("all_results", [])) > 1:
                with st.expander("Other matching locations"):
                    for r in geo_result["all_results"][1:]:
                        st.markdown(f"- {r['name']} ({r['lat']:.4f}, {r['lon']:.4f})")
        else:
            st.error(f"❌ Could not find '{place_query}'. Try a more specific name.")

    st.markdown("<div style='text-align:center; color:var(--muted); font-size:0.75rem; margin:0.3rem 0;'>─── OR select a preset ───</div>", unsafe_allow_html=True)

    PRESETS = {
        "Custom Location": (12.0, 76.5),
        "── South India ──": (11.66, 76.63),
        "🐯 Bandipur National Park (Karnataka)": (11.66, 76.63),
        "🐘 Nagarhole / Rajiv Gandhi NP (Karnataka)": (12.05, 76.15),
        "🦁 Mudumalai Tiger Reserve (Tamil Nadu)": (11.56, 76.55),
        "🌿 Wayanad Wildlife Sanctuary (Kerala)": (11.60, 76.02),
        "🐘 Periyar Tiger Reserve (Kerala)": (9.47, 77.17),
        "🐆 Sathyamangalam TR (Tamil Nadu)": (11.50, 77.25),
        "🌲 BR Hills Sanctuary (Karnataka)": (11.99, 77.16),
        "── Central India ──": (22.33, 80.62),
        "🐯 Kanha National Park (MP)": (22.33, 80.62),
        "🐘 Tadoba-Andhari Reserve (Maharashtra)": (20.20, 79.35),
        "🦁 Pench Tiger Reserve (MP)": (21.72, 79.30),
        "🌲 Satpura Tiger Reserve (MP)": (22.52, 78.12),
    }

    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        preset = st.selectbox("Predefined Wildlife Zones", list(PRESETS.keys()))
    with c2:
        default_lat, default_lon = PRESETS[preset]
        if resolved_lat is not None:
            default_lat = resolved_lat
        lat = st.number_input("Latitude", min_value=6.0, max_value=37.0, value=default_lat, step=0.01)
    with c3:
        if resolved_lon is not None:
            default_lon = resolved_lon
        lon = st.number_input("Longitude", min_value=68.0, max_value=97.0, value=default_lon, step=0.01)

    c4, c5, c6 = st.columns(3)
    with c4: speed_limit  = st.selectbox("Speed Limit (km/h)", [30, 40, 60, 80, 100], index=2)
    with c5: actual_speed = st.slider("Actual Speed (km/h)", 10, 140, 65)
    with c6: past_acc     = st.slider("Past Accident Count", 0, 30, 3)

    run_btn = st.button("⚡ EXTRACT DATA & PREDICT RISK", use_container_width=True)

    if run_btn:
        # ── RUN THE PIPELINE ──────────────────────────────────────────────────
        with st.spinner("📡 Extracting real-time data from APIs, news sites & govt portals…"):
            extractor = RealtimeDataExtractor()
            feature_df = extractor.extract_all(
                lat, lon,
                speed_limit=speed_limit,
                actual_speed=actual_speed,
                past_accidents=past_acc,
            )
            extraction_log = extractor.get_extraction_log()
            summary        = extractor.get_extraction_summary()

        # Save to session_state so Map page can access it
        spatial_data = None
        for er in extractor.extraction_log:
            if er.source_type == "computation" and "nearest_pa" in er.data:
                spatial_data = er.data
        st.session_state["last_prediction"] = {
            "lat": lat, "lon": lon,
            "location_name": resolved_name or preset,
            "features": feature_df.iloc[0].to_dict(),
            "log": extraction_log,
            "summary": summary,
            "spatial": spatial_data,
        }

        # ── DATA SOURCE STATUS ────────────────────────────────────────────────
        st.markdown("<div class='section-header'>📡 Real-Time Data Extraction — 7 Sources</div>", unsafe_allow_html=True)

        # Group by source type for display
        api_sources  = [l for l in extraction_log if l.get('source_type') == 'api']
        news_sources = [l for l in extraction_log if l.get('source_type') == 'news']
        govt_sources = [l for l in extraction_log if l.get('source_type') == 'government']
        comp_sources = [l for l in extraction_log if l.get('source_type') == 'computation']

        # Row 1: APIs
        st.markdown("**🌐 API Sources**", unsafe_allow_html=True)
        api_cols = st.columns(max(len(api_sources), 1))
        for i, log_entry in enumerate(api_sources):
            with api_cols[i]:
                si = "✅" if log_entry["status"]=="success" else "⚠️"
                fs = ", ".join(log_entry["features"][:3])
                st.markdown(f"<div class='source-card'><div class='source-name'>{si} {log_entry['source']}</div><div class='source-url'>{log_entry['api_url'][:70]}…</div><div style='font-size:0.72rem;'><span class='status-{log_entry["status"]}'>{log_entry['status'].upper()}</span> · {log_entry['response_ms']}ms</div><div class='source-features'>→ {fs}</div></div>", unsafe_allow_html=True)

        # Row 2: News + Govt
        ng_col1, ng_col2 = st.columns(2)
        for col, label, sources in [(ng_col1, "📰 News Websites", news_sources), (ng_col2, "🏛️ Government Portals", govt_sources)]:
            with col:
                st.markdown(f"**{label}**")
                for log_entry in sources:
                    si = "✅" if log_entry["status"]=="success" else "⚠️"
                    fs = ", ".join(str(f) for f in log_entry["features"][:3])
                    st.markdown(f"<div class='source-card'><div class='source-name'>{si} {log_entry['source']}</div><div style='font-size:0.72rem;'><span class='status-{log_entry["status"]}'>{log_entry['status'].upper()}</span> · {log_entry['response_ms']}ms</div><div class='source-features'>→ {fs}</div></div>", unsafe_allow_html=True)

        # Row 3: Computations
        if comp_sources:
            st.markdown("**🧮 Computed Sources**")
            comp_cols = st.columns(len(comp_sources))
            for i, log_entry in enumerate(comp_sources):
                with comp_cols[i]:
                    st.markdown(f"<div class='source-card'><div class='source-name'>✅ {log_entry['source']}</div><div style='font-size:0.72rem;'>SUCCESS · {log_entry['response_ms']}ms</div><div class='source-features'>→ {', '.join(log_entry['features'][:4])}</div></div>", unsafe_allow_html=True)

        st.plotly_chart(data_source_status_chart(extraction_log), use_container_width=True)

        # ── NEWS ARTICLES ─────────────────────────────────────────────────────
        news_data = None
        govt_detail = None
        for log_entry in extraction_log:
            if log_entry.get('source_type') == 'news':
                # Get news data from extractor
                for er in extractor.extraction_log:
                    if er.source_type == 'news':
                        news_data = er.data
            if log_entry.get('source_type') == 'government':
                for er in extractor.extraction_log:
                    if er.source_type == 'government':
                        govt_detail = er.data

        if news_data and news_data.get('top_articles'):
            st.markdown("<div class='section-header'>📰 Recent Wildlife News from Indian Media</div>", unsafe_allow_html=True)
            nc1, nc2 = st.columns([1, 1])
            with nc1:
                st.markdown(f"<div class='metric-card'><h3>Wildlife Articles Found</h3><div class='value' style='font-size:1.4rem;'>{news_data.get('total_wildlife_articles',0)}</div><div class='sub'>From The Hindu, NDTV, Down to Earth, Google News</div></div>", unsafe_allow_html=True)
            with nc2:
                st.markdown(f"<div class='metric-card'><h3>South India Mentions</h3><div class='value' style='font-size:1.4rem;'>{news_data.get('south_india_articles',0)}</div><div class='sub'>Matching Karnataka, Kerala, Tamil Nadu</div></div>", unsafe_allow_html=True)
            for art in news_data['top_articles'][:5]:
                si_tag = " 🟢 South India" if art.get('south_india_match') else ""
                sp_tag = f" | Species: {', '.join(art.get('species_mentioned',[])) }" if art.get('species_mentioned') else ""
                st.markdown(f"- **[{art['source']}]** {art['title']}{si_tag}{sp_tag}")

        if govt_detail:
            st.markdown("<div class='section-header'>🏛️ Indian Government Portal Status</div>", unsafe_allow_html=True)
            gs = govt_detail.get('source_status', {})
            gc1, gc2 = st.columns(2)
            with gc1:
                for name, status in list(gs.items())[:4]:
                    st.markdown(f"- **{name}**: {status}")
            with gc2:
                for name, status in list(gs.items())[4:]:
                    st.markdown(f"- **{name}**: {status}")
            st.markdown(f"📍 **Nearest State Dept**: {govt_detail.get('nearest_state_dept','')} ({govt_detail.get('nearest_state_url','')})")

        # ── PREDICTION RESULTS ────────────────────────────────────────────────
        st.markdown("<div class='section-header'>🤖 Dual-Model Prediction Results</div>", unsafe_allow_html=True)

        if model.rf_model is not None:
            results = model.predict_both(feature_df)
            xgb_prob = results["xgb_probability"]
            rf_prob  = results["rf_probability"]
            avg_prob = results["avg_probability"]
            agree    = results["agreement"]
        else:
            xgb_prob = float(model.predict_risk(feature_df)[0])
            rf_prob  = xgb_prob
            avg_prob = xgb_prob
            agree    = True

        rl_xgb = get_risk_level(xgb_prob)
        rl_rf  = get_risk_level(rf_prob)
        rl_avg = get_risk_level(avg_prob)

        # Gauges
        st.plotly_chart(dual_risk_gauge(xgb_prob, rf_prob), use_container_width=True)

        # Risk badges
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div style='text-align:center;'>
              <span class='risk-badge' style='background:{rl_xgb["color"]}33; color:{rl_xgb["color"]}; border:1.5px solid {rl_xgb["color"]};'>
                {rl_xgb["emoji"]} XGBoost: {xgb_prob:.1%}
              </span>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div style='text-align:center;'>
              <span class='risk-badge' style='background:{rl_rf["color"]}33; color:{rl_rf["color"]}; border:1.5px solid {rl_rf["color"]};'>
                {rl_rf["emoji"]} Random Forest: {rf_prob:.1%}
              </span>
            </div>""", unsafe_allow_html=True)
        with r3:
            agree_str = "✅ Models Agree" if agree else "⚠️ Divergence Detected"
            st.markdown(f"""
            <div style='text-align:center;'>
              <span class='risk-badge' style='background:{rl_avg["color"]}33; color:{rl_avg["color"]}; border:1.5px solid {rl_avg["color"]};'>
                {rl_avg["emoji"]} Ensemble: {avg_prob:.1%} · {agree_str}
              </span>
            </div>""", unsafe_allow_html=True)

        # ── SHAP explanation ──────────────────────────────────────────────────
        st.markdown("<div class='section-header'>🔬 SHAP Explanation (XGBoost)</div>", unsafe_allow_html=True)
        try:
            sv, X_in, base = model.predict_shap(feature_df)
            sv_flat = sv[0] if sv.ndim == 2 else sv
            st.plotly_chart(
                shap_waterfall(sv_flat, model.feature_cols,
                               X_in.iloc[0].values, base),
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

        # ── Extracted feature table ───────────────────────────────────────────
        st.markdown("<div class='section-header'>📋 Extracted Feature Vector</div>", unsafe_allow_html=True)
        feat_display = feature_df.T.reset_index()
        feat_display.columns = ["Feature", "Value"]
        st.dataframe(feat_display, use_container_width=True, height=400)

        # ── Recommendations ──────────────────────────────────────────────────
        st.markdown("<div class='section-header'>🛡 Mitigation Recommendations</div>", unsafe_allow_html=True)
        recs = []
        row = feature_df.iloc[0]
        if row.get("night_flag", 0):     recs.append("🌙 **Nocturnal alert zone** — deploy flashing warning signs between 20:00–06:00")
        if row.get("ndvi", 0) > 0.6:    recs.append("🌲 **High vegetation corridor** — install wildlife detection sensors")
        if row.get("dist_water_km", 99) < 1: recs.append("💧 **Water source proximity** — install water crossing structures / underpasses")
        if row.get("speed_ratio", 0) > 1.1:  recs.append("🚗 **Over-speed detected** — enforce speed cameras and lower limit")
        if row.get("rainfall_mm", 0) > 30:   recs.append("🌧 **Low visibility conditions** — dynamic variable speed limits")
        if row.get("breeding_season", 0):     recs.append("🔥 **Breeding season** — temporary speed restrictions May–October")
        if row.get("corridor_dist_km", 99) < 2: recs.append("🗺 **Corridor proximity** — install wildlife fencing and crossing")
        if not recs: recs.append("✅ Risk factors within acceptable range — standard monitoring advised")
        for rec in recs:
            st.markdown(f"- {rec}")

        # ── Data source detail expanders ──────────────────────────────────────
        st.markdown("<div class='section-header'>📡 Detailed Data Source Logs</div>", unsafe_allow_html=True)
        for log_entry in extraction_log:
            status_icon = "✅" if log_entry["status"] == "success" else "⚠️" if log_entry["status"] == "fallback" else "❌"
            with st.expander(f"{status_icon} {log_entry['source']}  —  {log_entry['response_ms']}ms"):
                st.markdown(f"**API URL:** `{log_entry['api_url']}`")
                st.markdown(f"**Description:** {log_entry['description']}")
                st.markdown(f"**Status:** `{log_entry['status']}`")
                st.markdown(f"**Timestamp:** `{log_entry['timestamp']}`")
                st.markdown(f"**Features extracted:** {', '.join(log_entry['features'])}")
                if log_entry["error"]:
                    st.error(f"Error: {log_entry['error']}")
                if log_entry["raw_preview"]:
                    st.code(log_entry["raw_preview"], language="json")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RISK MAP (Heatmap + Live Prediction Toggle)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗺 Risk Map":
    from streamlit_folium import st_folium
    from utils.helpers import build_folium_map, get_risk_level
    from data.realtime_extractor import PROTECTED_AREAS, WILDLIFE_CORRIDORS
    import folium
    from folium.plugins import HeatMap, MarkerCluster

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 0.5rem 0;'>
      🗺 Geospatial <span style='color:#FF6B35;'>Risk Map</span>
    </h1>
    """, unsafe_allow_html=True)

    # ── Toggle: Full Heatmap vs Live Prediction ──────────────────────────────
    has_prediction = "last_prediction" in st.session_state
    toggle_options = ["🌡️ Full Risk Heatmap (All Data)", "📍 Live Prediction Result"]

    map_mode = st.radio(
        "Map View",
        toggle_options,
        index=1 if has_prediction else 0,
        horizontal=True,
        key="map_mode_toggle",
    )

    # ══════════════════════════════════════════════════════════════════════════
    #  MODE 1: FULL HEATMAP (training data + PAs + corridors)
    # ══════════════════════════════════════════════════════════════════════════
    if map_mode == toggle_options[0]:
        st.markdown("""
        <p style='color:#8B949E; font-size:0.82rem; margin:0 0 1rem 0;'>
          Risk heatmap from training data — density of wildlife-vehicle collision zones with all protected areas & corridors
        </p>
        """, unsafe_allow_html=True)

        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1: f_season  = st.multiselect("Season",    df["season"].unique().tolist(),    default=df["season"].unique().tolist())
        with fc2: f_species = st.multiselect("Species",   df["species"].unique().tolist(),   default=df["species"].unique().tolist())
        with fc3: f_road    = st.multiselect("Road Type", df["road_type"].unique().tolist(), default=df["road_type"].unique().tolist())

        map_df = df[df["season"].isin(f_season) & df["species"].isin(f_species) & df["road_type"].isin(f_road)]

        # Build rich Folium map
        m = folium.Map(location=[15.0, 77.5], zoom_start=6, tiles="CartoDB dark_matter", control_scale=True)

        # Heatmap layer from training data
        acc = map_df[map_df["accident"] == 1][["latitude", "longitude", "risk_score"]].dropna()
        heat_data = [[r.latitude, r.longitude, r.risk_score] for _, r in acc.iterrows()]
        if heat_data:
            HeatMap(
                heat_data, min_opacity=0.3, max_zoom=14, radius=18, blur=15,
                gradient={"0.2": "#06D6A0", "0.5": "#FFD166", "0.75": "#EF476F", "1.0": "#B5179E"},
            ).add_to(m)

        # Protected Areas layer
        pa_group = folium.FeatureGroup(name="🟢 Protected Areas (21)")
        for pa in PROTECTED_AREAS:
            folium.CircleMarker(
                location=[pa["lat"], pa["lon"]], radius=7,
                color="#06D6A0", fill=True, fill_color="#06D6A0", fill_opacity=0.7,
                popup=folium.Popup(f"<b>🟢 {pa['name']}</b><br>{pa.get('state','')}", max_width=220),
                tooltip=pa["name"],
            ).add_to(pa_group)
        pa_group.add_to(m)

        # Corridors layer
        cor_group = folium.FeatureGroup(name="🟠 Wildlife Corridors (11)")
        for cor in WILDLIFE_CORRIDORS:
            folium.CircleMarker(
                location=[cor["lat"], cor["lon"]], radius=5,
                color="#FFD166", fill=True, fill_color="#FFD166", fill_opacity=0.7,
                popup=folium.Popup(f"<b>🟠 {cor['name']}</b><br>Wildlife Corridor", max_width=220),
                tooltip=cor["name"],
            ).add_to(cor_group)
        cor_group.add_to(m)

        # High-risk markers
        high = map_df[map_df["risk_score"] > 0.70].head(150)
        cluster = MarkerCluster(name="🔴 High Risk Incidents").add_to(m)
        for _, row in high.iterrows():
            rl = get_risk_level(row["risk_score"])
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]], radius=5,
                color=rl["color"], fill=True, fill_opacity=0.8,
                popup=folium.Popup(
                    f"<b>{rl['emoji']} {rl['label']}</b><br>"
                    f"Risk: {row['risk_score']:.2%}<br>"
                    f"Species: {row.get('species','–')}<br>"
                    f"Road: {row.get('road_type','–')}<br>"
                    f"Hour: {int(row.get('hour',0)):02d}:00",
                    max_width=200),
            ).add_to(cluster)

        folium.LayerControl().add_to(m)

        # Render
        c_left, c_right = st.columns([3, 1])
        with c_left:
            st_folium(m, width=None, height=620, returned_objects=[])

        with c_right:
            st.markdown("<div class='section-header'>Map Legend</div>", unsafe_allow_html=True)
            for color, label in [("#06D6A0","🟢 Protected Areas"),("#FFD166","🟠 Corridors"),("#EF476F","🔴 High Risk"),("#B5179E","🟣 Critical")]:
                st.markdown(f"<div style='display:flex;align-items:center;gap:0.5rem;margin:0.3rem 0;'><div style='width:14px;height:14px;border-radius:50%;background:{color};'></div><span style='font-size:0.82rem;'>{label}</span></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='margin-top:0.5rem; padding:0.3rem 0; border-top:1px solid var(--border);'></div>
            <div class='section-header'>Summary</div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class='metric-card'>
              <h3>Records</h3><div class='value' style='font-size:1.3rem;'>{len(map_df):,}</div>
            </div>
            <div class='metric-card'>
              <h3>Accidents</h3><div class='value' style='font-size:1.3rem;'>{int(map_df["accident"].sum()):,}</div>
            </div>
            <div class='metric-card'>
              <h3>Avg Risk</h3><div class='value' style='font-size:1.3rem;'>{map_df["risk_score"].mean():.3f}</div>
            </div>
            <div class='metric-card'>
              <h3>Heatmap Points</h3><div class='value' style='font-size:1.3rem;'>{len(heat_data):,}</div>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  MODE 2: LIVE PREDICTION (from Live Risk Predictor result)
    # ══════════════════════════════════════════════════════════════════════════
    else:
        if not has_prediction:
            st.warning("⚠️ No live prediction data yet. Go to **🔮 Live Risk Predictor** and run an extraction first.")
            st.info("The Live Risk Predictor fetches real-time data from 7 sources (APIs, news, govt portals) and the results will appear here on the map.")
        else:
            pred = st.session_state["last_prediction"]
            feat = pred["features"]
            spatial = pred["spatial"] or {}
            pred_lat, pred_lon = pred["lat"], pred["lon"]
            loc_name = pred.get("location_name", "Searched Location")

            st.markdown(f"""
            <p style='color:#8B949E; font-size:0.82rem; margin:0 0 0.5rem 0;'>
              Showing extracted data for <b style='color:#FF6B35;'>{loc_name}</b> · Lat {pred_lat:.4f}, Lon {pred_lon:.4f}
            </p>
            """, unsafe_allow_html=True)

            # Build focused map
            m = folium.Map(location=[pred_lat, pred_lon], zoom_start=11, tiles="CartoDB dark_matter", control_scale=True)

            # Nearby PAs (within ~100km)
            from data.realtime_extractor import _haversine
            nearby_pas = []
            for pa in PROTECTED_AREAS:
                d = _haversine(pred_lat, pred_lon, pa["lat"], pa["lon"])
                if d < 100:
                    nearby_pas.append((d, pa))
                    folium.CircleMarker(
                        location=[pa["lat"], pa["lon"]], radius=8,
                        color="#06D6A0", fill=True, fill_color="#06D6A0", fill_opacity=0.7,
                        popup=folium.Popup(f"<b>🟢 {pa['name']}</b><br>{pa.get('state','')}<br>{d:.1f}km away", max_width=220),
                        tooltip=f"{pa['name']} ({d:.0f}km)",
                    ).add_to(m)

            # Nearby corridors
            nearby_cors = []
            for cor in WILDLIFE_CORRIDORS:
                d = _haversine(pred_lat, pred_lon, cor["lat"], cor["lon"])
                if d < 100:
                    nearby_cors.append((d, cor))
                    folium.CircleMarker(
                        location=[cor["lat"], cor["lon"]], radius=6,
                        color="#FFD166", fill=True, fill_color="#FFD166", fill_opacity=0.7,
                        popup=folium.Popup(f"<b>🟠 {cor['name']}</b><br>{d:.1f}km away", max_width=220),
                        tooltip=f"{cor['name']} ({d:.0f}km)",
                    ).add_to(m)

            # Prediction marker
            risk_score = feat.get("driver_risk", 0.5)
            risk_color = "#06D6A0" if risk_score < 0.4 else "#FFD166" if risk_score < 0.7 else "#EF476F" if risk_score < 1.0 else "#B5179E"

            popup_html = f"""
            <div style='font-family:sans-serif; min-width:260px;'>
                <h3 style='margin:0 0 6px 0; color:{risk_color};'>📍 {loc_name}</h3>
                <table style='font-size:11px; width:100%;'>
                    <tr><td><b>Species</b></td><td>{feat.get('species','—').capitalize()} (risk {feat.get('species_risk',0):.0%})</td></tr>
                    <tr><td><b>Road Type</b></td><td>{feat.get('road_type','—')}</td></tr>
                    <tr><td><b>Temperature</b></td><td>{feat.get('temperature_c',0):.1f}°C</td></tr>
                    <tr><td><b>Humidity</b></td><td>{feat.get('humidity_pct',0):.0f}%</td></tr>
                    <tr><td><b>Rain</b></td><td>{feat.get('rainfall_mm',0):.1f}mm</td></tr>
                    <tr><td><b>Visibility</b></td><td>{feat.get('visibility_m',0)}m</td></tr>
                    <tr><td><b>NDVI</b></td><td>{feat.get('ndvi',0):.3f}</td></tr>
                    <tr><td><b>Nearest PA</b></td><td>{spatial.get('nearest_pa','—')} ({spatial.get('protected_dist_km',0):.1f}km)</td></tr>
                    <tr><td><b>Nearest Corridor</b></td><td>{spatial.get('nearest_corridor','—')} ({spatial.get('corridor_dist_km',0):.1f}km)</td></tr>
                    <tr><td><b>Driver Risk</b></td><td style='color:{risk_color}; font-weight:bold;'>{risk_score:.3f}</td></tr>
                </table>
            </div>
            """
            folium.Marker(
                location=[pred_lat, pred_lon],
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"📍 {loc_name} — Risk {risk_score:.3f}",
                icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
            ).add_to(m)

            # Analysis radius
            folium.Circle(
                location=[pred_lat, pred_lon], radius=5000,
                color=risk_color, fill=True, fill_opacity=0.08, popup="5km analysis area",
            ).add_to(m)

            # Risk heatmap point (single-point, visually showing the searched area)
            HeatMap(
                [[pred_lat, pred_lon, risk_score]],
                min_opacity=0.4, radius=40, blur=25,
                gradient={"0.2": "#06D6A0", "0.5": "#FFD166", "0.75": "#EF476F", "1.0": "#B5179E"},
            ).add_to(m)

            folium.LayerControl().add_to(m)

            # Render
            c_left, c_right = st.columns([3, 1])
            with c_left:
                st_folium(m, width=None, height=620, returned_objects=[])

            with c_right:
                st.markdown("<div class='section-header'>📍 Extracted Data</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='metric-card'>
                  <h3>Location</h3><div class='value' style='font-size:0.85rem;'>{loc_name[:40]}</div>
                  <div class='sub'>{pred_lat:.4f}, {pred_lon:.4f}</div>
                </div>
                <div class='metric-card'>
                  <h3>Nearest PA</h3><div class='value' style='font-size:0.9rem;'>{spatial.get('nearest_pa','—')}</div>
                  <div class='sub'>{spatial.get('protected_dist_km',0):.1f}km · {spatial.get('nearest_pa_state','')}</div>
                </div>
                <div class='metric-card'>
                  <h3>Nearest Corridor</h3><div class='value' style='font-size:0.9rem;'>{spatial.get('nearest_corridor','—')}</div>
                  <div class='sub'>{spatial.get('corridor_dist_km',0):.1f}km</div>
                </div>
                <div class='metric-card'>
                  <h3>Species</h3><div class='value' style='font-size:1.1rem;'>{feat.get('species','—').capitalize()}</div>
                  <div class='sub'>Risk: {feat.get('species_risk',0):.0%}</div>
                </div>
                <div class='metric-card'>
                  <h3>Weather</h3><div class='value' style='font-size:0.9rem;'>{feat.get('temperature_c',0):.1f}°C · {feat.get('humidity_pct',0):.0f}%</div>
                  <div class='sub'>Rain {feat.get('rainfall_mm',0):.1f}mm · Vis {feat.get('visibility_m',0)}m</div>
                </div>
                <div class='metric-card'>
                  <h3>NDVI / Road</h3><div class='value' style='font-size:0.9rem;'>{feat.get('ndvi',0):.3f} · {feat.get('road_type','—')}</div>
                  <div class='sub'>Width {feat.get('road_width_m',0)}m · Light {'Yes' if feat.get('street_lighting',0) else 'No'}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"<div style='margin-top:0.5rem; border-top:1px solid var(--border); padding-top:0.4rem;'></div>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>📡 Data Sources</div>", unsafe_allow_html=True)
                summary = pred["summary"]
                st.markdown(f"**{summary['success_count']}/{summary['total_sources']}** sources · **{summary['total_time_ms']}ms**")
                for l in pred["log"]:
                    icon = "✅" if l["status"]=="success" else "⚠️"
                    st.markdown(f"<div style='font-size:0.68rem;'>{icon} {l['source'][:28]} · {l['response_ms']}ms</div>", unsafe_allow_html=True)

                st.markdown(f"""
                <div style='margin-top:0.5rem; border-top:1px solid var(--border); padding-top:0.4rem;'></div>
                <div class='section-header'>📌 Nearby</div>
                """, unsafe_allow_html=True)
                st.markdown(f"**{len(nearby_pas)}** Protected Areas within 100km")
                st.markdown(f"**{len(nearby_cors)}** Corridors within 100km")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    from utils.helpers import PALETTE, PLOTLY_LAYOUT

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 1.2rem 0;'>
      📊 Exploratory <span style='color:#FF6B35;'>Analytics</span>
    </h1>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🕐 Temporal", "🌿 Environmental", "🚗 Traffic", "🦁 Wildlife", "📉 Historical"
    ])

    _AX = dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            pivot_hd = df.pivot_table(index="hour", columns="day_of_week",
                                      values="accident", aggfunc="mean")
            days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            fig = go.Figure(go.Heatmap(
                z=pivot_hd.values, x=days, y=list(range(24)),
                colorscale=[[0,"#0a1628"],[0.5,"#EF476F"],[1,"#B5179E"]],
                showscale=True,
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title="Accident Rate: Hour × Day", height=420)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            grp = df.groupby("season")["accident"].mean().sort_values(ascending=False)
            fig2 = go.Figure(go.Bar(
                x=grp.index, y=grp.values,
                marker_color=[PALETTE["critical"],PALETTE["high"],PALETTE["medium"],PALETTE["low"]],
                text=[f"{v:.1%}" for v in grp.values], textposition="outside",
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, title="Accident Rate by Season", yaxis_tickformat=".0%", height=360)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.histogram(df, x="ndvi", color="season", nbins=40,
                                barmode="overlay", opacity=0.7,
                                color_discrete_sequence=[PALETTE["accent2"],PALETTE["accent1"],PALETTE["accent3"],PALETTE["high"]],
                                title="NDVI Distribution by Season")
            fig3.update_layout(**PLOTLY_LAYOUT, height=360)
            st.plotly_chart(fig3, use_container_width=True)
        with col4:
            fig4 = px.scatter(df.sample(2000, random_state=1),
                              x="rainfall_mm", y="visibility_m",
                              color="accident", opacity=0.5, size_max=6,
                              color_continuous_scale=[[0,PALETTE["accent2"]],[1,PALETTE["high"]]],
                              title="Rainfall vs Visibility (coloured by accident)")
            fig4.update_layout(**PLOTLY_LAYOUT, height=360)
            st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        col5, col6 = st.columns(2)
        with col5:
            grp_speed = df.copy()
            grp_speed["speed_bin"] = pd.cut(grp_speed["speed_ratio"], bins=10)
            sr = grp_speed.groupby("speed_bin", observed=True)["accident"].mean()
            fig5 = go.Figure(go.Bar(
                x=[str(b) for b in sr.index], y=sr.values,
                marker_color=PALETTE["accent1"],
                text=[f"{v:.1%}" for v in sr.values], textposition="outside",
            ))
            fig5.update_layout(**PLOTLY_LAYOUT, title="Accident Rate by Speed Ratio", height=360)
            st.plotly_chart(fig5, use_container_width=True)
        with col6:
            rt = df.groupby("road_type")["accident"].agg(["mean","count"]).reset_index().sort_values("mean")
            fig6 = go.Figure(go.Bar(
                x=rt["mean"], y=rt["road_type"],
                orientation="h",
                marker_color=PALETTE["accent3"],
                text=[f"{v:.1%}" for v in rt["mean"]], textposition="outside",
            ))
            fig6.update_layout(**PLOTLY_LAYOUT, title="Accident Rate by Road Type", height=360)
            st.plotly_chart(fig6, use_container_width=True)

    with tab4:
        col7, col8 = st.columns(2)
        with col7:
            sp = df.groupby("species")["risk_score"].describe()[["mean","50%","max"]].reset_index()
            fig7 = px.bar(sp, x="species", y="mean", error_y=sp["max"]-sp["mean"],
                          color="mean",
                          color_continuous_scale=[[0,PALETTE["low"]],[0.5,PALETTE["high"]],[1,PALETTE["critical"]]],
                          title="Mean Risk Score by Species")
            fig7.update_layout(**PLOTLY_LAYOUT, height=360)
            st.plotly_chart(fig7, use_container_width=True)
        with col8:
            crr = df[df["accident"]==1].groupby(["species","road_type"]).size().reset_index(name="count")
            fig8 = px.treemap(crr, path=["species","road_type"], values="count",
                              color="count",
                              color_continuous_scale=[[0,"#0d1117"],[1,"#FF6B35"]],
                              title="Accident Distribution: Species × Road")
            fig8.update_layout(**PLOTLY_LAYOUT, height=380)
            st.plotly_chart(fig8, use_container_width=True)

    with tab5:
        col9, col10 = st.columns(2)
        with col9:
            fig9 = px.box(df, x="road_type", y="past_accidents",
                          color="season",
                          color_discrete_sequence=[PALETTE["accent2"],PALETTE["accent3"],PALETTE["high"],PALETTE["critical"]],
                          title="Past Accidents Distribution by Road & Season")
            fig9.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig9, use_container_width=True)
        with col10:
            fig10 = px.scatter(df.sample(3000, random_state=2),
                               x="kde_density", y="risk_score",
                               color="road_type", opacity=0.6, size_max=8,
                               title="KDE Density vs Risk Score")
            fig10.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig10, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4.5 — FUTURE HOTSPOT FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Future Hotspots":
    from models.train import WildlifeRiskModel, FEATURE_GROUPS
    from data.generate_data import ALL_HIGHWAY_SEGMENTS
    import shap

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 0.3rem 0;'>
      ⚡ Future Hotspot <span style='color:#FF6B35;'>Forecasting</span>
    </h1>
    <p style='color:#8B949E; font-size:0.82rem; margin:0 0 1.2rem 0;'>
      Identifies highway segments with <b>low historical incidents but high predicted risk</b> —
      emerging danger zones where preventive infrastructure should be deployed before accidents happen
    </p>
    """, unsafe_allow_html=True)

    # ── Segment-level analysis ────────────────────────────────────────────────
    if 'highway_segment' not in df.columns:
        st.warning("Dataset does not contain highway segment data. Please retrain with the updated data generator.")
    else:
        # Aggregate by highway segment
        seg_stats = df.groupby('highway_segment').agg(
            total_records=('accident', 'count'),
            total_accidents=('accident', 'sum'),
            accident_rate=('accident', 'mean'),
            avg_risk_score=('risk_score', 'mean'),
            avg_movement=('movement_score', 'mean'),
            avg_driver_risk=('driver_risk', 'mean'),
            avg_ndvi=('ndvi', 'mean'),
            avg_corridor_dist=('corridor_dist_km', 'mean'),
            avg_visibility=('visibility_m', 'mean'),
            avg_speed_ratio=('speed_ratio', 'mean'),
            lat_center=('latitude', 'mean'),
            lon_center=('longitude', 'mean'),
        ).reset_index()

        # Predict risk probability for each segment's average conditions
        seg_stats['predicted_risk'] = seg_stats['avg_risk_score']

        # Future hotspots: LOW historical accident rate BUT HIGH predicted characteristics
        seg_stats['hotspot_score'] = (
            seg_stats['avg_risk_score'] * 0.4 +
            seg_stats['avg_movement'] * 0.2 +
            seg_stats['avg_driver_risk'] / 3 * 0.2 +
            (1 - seg_stats['accident_rate']) * 0.2  # Bonus for low historical rate
        )

        # Flag: "Emerging" = low history + high features
        median_acc = seg_stats['accident_rate'].median()
        median_risk = seg_stats['hotspot_score'].median()
        seg_stats['category'] = seg_stats.apply(
            lambda r: '⚠️ EMERGING HOTSPOT' if r['accident_rate'] < median_acc and r['hotspot_score'] > median_risk
            else '🔴 Known High Risk' if r['accident_rate'] >= median_acc and r['hotspot_score'] > median_risk
            else '🟢 Low Risk' if r['accident_rate'] < median_acc
            else '🟡 Moderate', axis=1
        )

        # Sort by hotspot score
        seg_stats = seg_stats.sort_values('hotspot_score', ascending=False)

        # ── KPI Cards ────────────────────────────────────────────────────────
        emerging = seg_stats[seg_stats['category'] == '⚠️ EMERGING HOTSPOT']
        known_high = seg_stats[seg_stats['category'] == '🔴 Known High Risk']

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""<div class='metric-card'>
              <h3>Total Segments</h3><div class='value'>{len(seg_stats)}</div>
              <div class='sub'>Analyzed</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class='metric-card'>
              <h3>⚠️ Emerging Hotspots</h3><div class='value' style='color:#FFD166;'>{len(emerging)}</div>
              <div class='sub'>Low history, high risk</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class='metric-card'>
              <h3>🔴 Known High Risk</h3><div class='value' style='color:#EF476F;'>{len(known_high)}</div>
              <div class='sub'>High history + risk</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class='metric-card'>
              <h3>Top Hotspot Score</h3><div class='value' style='color:#FF6B35;'>{seg_stats['hotspot_score'].max():.3f}</div>
              <div class='sub'>{seg_stats.iloc[0]['highway_segment'][:25]}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Emerging Hotspots Detail ─────────────────────────────────────────
        st.markdown("<div class='section-header'>⚠️ Emerging Hotspots — Where to Deploy Prevention Infrastructure</div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color:#8B949E; font-size:0.8rem;'>
          These segments have <b>few historical accidents</b> but share <b>high-risk environmental &
          infrastructural characteristics</b> with known danger zones. They represent
          <b>future collision hotspots</b> where preventive barriers, animal crossings, or speed controls
          should be installed <i>before</i> incidents escalate.
        </p>
        """, unsafe_allow_html=True)

        for i, row in emerging.head(5).iterrows():
            # Find matching highway segment info
            seg_info = next((s for s in ALL_HIGHWAY_SEGMENTS if s["name"] == row["highway_segment"]), {})
            state = seg_info.get("state", "—")

            # Determine risk factors (plain language for officials)
            risk_factors = []
            if row['avg_ndvi'] > 0.5:
                risk_factors.append("🌿 Dense vegetation (animals cross frequently)")
            if row['avg_corridor_dist'] < 3:
                risk_factors.append("🦁 Very close to wildlife corridor")
            if row['avg_speed_ratio'] > 1.0:
                risk_factors.append("🚗 Vehicles exceed speed limits")
            if row['avg_visibility'] < 500:
                risk_factors.append("🌫️ Poor visibility conditions")
            if row['avg_movement'] > 0.35:
                risk_factors.append("🐾 High animal activity score")
            if row['avg_driver_risk'] > 0.7:
                risk_factors.append("⚡ High driver risk index")
            if not risk_factors:
                risk_factors.append("📊 Combined risk factors above threshold")

            st.markdown(f"""
            <div style='background:linear-gradient(135deg, rgba(255,209,102,0.08), rgba(255,107,53,0.05));
                        border:1px solid rgba(255,209,102,0.3); border-radius:12px;
                        padding:1.2rem; margin:0.6rem 0;'>
              <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;'>
                <div>
                  <span style='font-size:1.1rem; font-weight:600; color:#FFD166;'>⚠️ {row['highway_segment']}</span>
                  <span style='font-size:0.75rem; color:#8B949E; margin-left:0.5rem;'>{state}</span>
                </div>
                <div style='text-align:right;'>
                  <div style='font-size:1.2rem; font-weight:700; color:#FF6B35;'>Hotspot Score: {row['hotspot_score']:.3f}</div>
                  <div style='font-size:0.7rem; color:#8B949E;'>Accident Rate: {row['accident_rate']:.1%} (low history)</div>
                </div>
              </div>
              <div style='display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:0.5rem; font-size:0.75rem; color:#E6EDF3;'>
                <div>🌿 NDVI: <b>{row['avg_ndvi']:.3f}</b></div>
                <div>🦁 Corridor: <b>{row['avg_corridor_dist']:.1f}km</b></div>
                <div>🐾 Movement: <b>{row['avg_movement']:.3f}</b></div>
                <div>🚗 Speed Ratio: <b>{row['avg_speed_ratio']:.2f}</b></div>
              </div>
              <div style='margin-top:0.5rem; border-top:1px solid rgba(255,209,102,0.15); padding-top:0.5rem;'>
                <div style='font-size:0.72rem; color:#FFD166; font-weight:600; margin-bottom:0.3rem;'>
                  🔍 Why this segment is risky (for forest officials):
                </div>
                {''.join(f"<div style='font-size:0.72rem; color:#E6EDF3; margin:0.15rem 0;'>{rf}</div>" for rf in risk_factors)}
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Hotspot Comparison Chart ─────────────────────────────────────────
        st.markdown("<div class='section-header'>📊 Segment Risk vs. Historical Accident Rate</div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color:#8B949E; font-size:0.78rem;'>
          Segments in the <b>top-left quadrant</b> (low history, high risk) are emerging hotspots.
          Segments in the <b>top-right</b> are known danger zones.
        </p>
        """, unsafe_allow_html=True)

        color_map = {
            '⚠️ EMERGING HOTSPOT': '#FFD166',
            '🔴 Known High Risk': '#EF476F',
            '🟢 Low Risk': '#06D6A0',
            '🟡 Moderate': '#8B949E',
        }
        fig_scatter = go.Figure()
        for cat, color in color_map.items():
            subset = seg_stats[seg_stats['category'] == cat]
            if len(subset) > 0:
                fig_scatter.add_trace(go.Scatter(
                    x=subset['accident_rate'], y=subset['hotspot_score'],
                    mode='markers+text',
                    marker=dict(size=12, color=color, opacity=0.85, line=dict(width=1, color='white')),
                    text=subset['highway_segment'].str[:18],
                    textposition='top center',
                    textfont=dict(size=8, color=color),
                    name=cat,
                    hovertemplate='<b>%{text}</b><br>Accident Rate: %{x:.1%}<br>Hotspot Score: %{y:.3f}<extra></extra>',
                ))
        fig_scatter.add_vline(x=median_acc, line_dash="dash", line_color="#8B949E", opacity=0.5)
        fig_scatter.add_hline(y=median_risk, line_dash="dash", line_color="#8B949E", opacity=0.5)
        fig_scatter.update_layout(
            xaxis_title="Historical Accident Rate", yaxis_title="Hotspot Score (Predicted Risk)",
            template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
            font=dict(color="#E6EDF3"), height=500,
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # ── SHAP Global Feature Importance ───────────────────────────────────
        st.markdown("<div class='section-header'>🧠 SHAP — Global Feature Importance (What Drives Risk Overall)</div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color:#8B949E; font-size:0.78rem;'>
          SHAP (SHapley Additive exPlanations) decomposes the model's prediction into contributions
          from each feature. Below shows which features have the <b>most influence on risk predictions</b>
          across all highway segments. This tells forest officials where to focus intervention resources.
        </p>
        """, unsafe_allow_html=True)

        # Global SHAP bar chart
        if shap_imp:
            shap_df = pd.DataFrame(shap_imp).head(15)
            fig_shap = go.Figure(go.Bar(
                y=shap_df['feature'], x=shap_df['shap_mean'],
                orientation='h',
                marker=dict(
                    color=shap_df['shap_mean'],
                    colorscale=[[0, '#06D6A0'], [0.5, '#FFD166'], [1.0, '#EF476F']],
                    line=dict(width=1, color='rgba(255,255,255,0.1)'),
                ),
                hovertemplate='<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>',
            ))
            fig_shap.update_layout(
                template="plotly_dark", paper_bgcolor="#0D1117", plot_bgcolor="#161B22",
                font=dict(color="#E6EDF3"), height=450,
                xaxis_title="Mean |SHAP Value| (Impact on Prediction)",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=150),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            # Plain-language explanations for top features
            st.markdown("<div class='section-header'>📋 What This Means for Forest Officials</div>", unsafe_allow_html=True)
            explanations = {
                'movement_score': "**Animal Activity** — The composite animal movement score is the biggest predictor. Areas with high vegetation (NDVI), near water, during dawn/dusk, and in breeding season see the most crossings.",
                'driver_risk': "**Driver Behavior** — Speed violations, night driving, and driving on forest roads dramatically increase collision risk. Speed enforcement is the most actionable intervention.",
                'species_risk': "**Species Type** — Elephants and tigers cause the highest-impact collisions. Their movement patterns are seasonal and predictable.",
                'kde_density': "**Historical Density** — Past accident clusters strongly predict future ones. Known hotspots remain dangerous unless mitigated.",
                'road_type': "**Road Classification** — Forest roads and rural roads are much riskier than national highways, despite lower traffic. They lack barriers and lighting.",
                'night_flag': "**Night Driving** — Risk roughly doubles during nighttime (8 PM – 6 AM). Night-vision cameras and reflective fencing are effective countermeasures.",
                'corridor_dist_km': "**Corridor Distance** — Closer to wildlife corridors = higher risk. Underpasses/overpasses within 3km of corridors reduce mortality 60-80%.",
                'speed_ratio': "**Speed Compliance** — Drivers exceeding speed limits by even 20% dramatically increase stopping distance and collision severity.",
                'ndvi': "**Vegetation Density** — Dense roadside vegetation blocks sightlines. Strategic clearing of 5m strips along highways improves visibility.",
                'past_accidents': "**Repeat Locations** — Areas with 3+ past incidents are 4× more likely to see future ones. Prioritize these for infrastructure upgrades.",
                'rainfall_mm': "**Rainfall** — Wet roads + reduced visibility + animal movement to water create a triple threat during monsoon season.",
                'visibility_m': "**Visibility** — Fog, rain, and dense canopy reduce sighting distance. Electronic warning signs triggered by poor visibility save lives.",
            }
            top_feats = [s['feature'] for s in shap_imp[:6]]
            for feat in top_feats:
                expl = explanations.get(feat, f"**{feat}** — This feature contributes significantly to risk prediction.")
                st.markdown(f"- {expl}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Model Insights":
    from utils.helpers import (
        feature_importance_chart, rf_importance_chart,
        confusion_matrix_chart, model_comparison_chart,
        roc_comparison_chart, PALETTE, PLOTLY_LAYOUT
    )
    from models.train import FEATURE_GROUPS

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 1.2rem 0;'>
      🧠 Model <span style='color:#FF6B35;'>Insights & Explainability</span>
    </h1>
    """, unsafe_allow_html=True)

    _AX = dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"])

    tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs([
        "📈 Performance", "🔬 XGBoost SHAP", "🌲 RF Importance",
        "📊 SHAP Beeswarm", "⚙️ Feature Groups"
    ])

    with tab_a:
        # ── Metric cards ──────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>XGBoost Metrics</div>", unsafe_allow_html=True)
        xc1, xc2, xc3, xc4, xc5 = st.columns(5)
        for col, k, label in [
            (xc1, "roc_auc",       "ROC-AUC"),
            (xc2, "avg_precision", "Avg Precision"),
            (xc3, "accuracy",      "Accuracy"),
            (xc4, "brier_score",   "Brier Score"),
            (xc5, "cv_roc_auc_mean", "CV AUC (5-fold)"),
        ]:
            v = xgb_metrics.get(k, 0)
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                  <h3>{label}</h3>
                  <div class='value' style='font-size:1.5rem;'>{v:.4f}</div>
                </div>""", unsafe_allow_html=True)

        if rf_metrics:
            st.markdown("<div class='section-header'>Random Forest Metrics</div>", unsafe_allow_html=True)
            rc1, rc2, rc3, rc4, rc5 = st.columns(5)
            for col, k, label in [
                (rc1, "roc_auc",       "ROC-AUC"),
                (rc2, "avg_precision", "Avg Precision"),
                (rc3, "accuracy",      "Accuracy"),
                (rc4, "brier_score",   "Brier Score"),
                (rc5, "cv_roc_auc_mean", "CV AUC (5-fold)"),
            ]:
                v = rf_metrics.get(k, 0)
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                      <h3>{label}</h3>
                      <div class='value' style='font-size:1.5rem; color:#58A6FF;'>{v:.4f}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        cmp1, cmp2 = st.columns(2)
        with cmp1:
            if rf_metrics:
                st.plotly_chart(model_comparison_chart(xgb_metrics, rf_metrics), use_container_width=True)
        with cmp2:
            if rf_metrics:
                st.plotly_chart(roc_comparison_chart(xgb_metrics, rf_metrics), use_container_width=True)

        # Confusion matrices
        st.markdown("<br>", unsafe_allow_html=True)
        cm1, cm2 = st.columns(2)
        with cm1:
            st.plotly_chart(confusion_matrix_chart(
                xgb_metrics.get("confusion", [[0,0],[0,0]]), "XGBoost Confusion Matrix"
            ), use_container_width=True)
        with cm2:
            if rf_metrics:
                st.plotly_chart(confusion_matrix_chart(
                    rf_metrics.get("confusion", [[0,0],[0,0]]), "RF Confusion Matrix"
                ), use_container_width=True)

    with tab_b:
        if shap_imp:
            st.plotly_chart(feature_importance_chart(shap_imp), use_container_width=True)
        else:
            st.info("SHAP importance not available.")

    with tab_c:
        if rf_imp:
            st.plotly_chart(rf_importance_chart(rf_imp), use_container_width=True)
        else:
            st.info("Random Forest importance not available. Retrain the models.")

    with tab_d:
        st.markdown("<div class='section-header'>SHAP Beeswarm — Global Feature Impact</div>",
                    unsafe_allow_html=True)
        sv  = model.shap_values
        ssp = model.shap_sample
        if sv is not None and ssp is not None:
            top_n    = 15
            top_feat = [x["feature"] for x in sorted(shap_imp, key=lambda x: x["shap_mean"], reverse=True)[:top_n]]
            idxs     = [model.feature_cols.index(f) for f in top_feat if f in model.feature_cols]

            traces = []
            for rank, (fi, fname) in enumerate(zip(idxs, top_feat)):
                fv = ssp.iloc[:, fi].values
                fv_norm = (fv - fv.min()) / ((fv.max() - fv.min()) + 1e-9)
                jitter  = np.random.uniform(-0.25, 0.25, len(sv))
                traces.append(go.Scatter(
                    x    = sv[:, fi],
                    y    = [rank + j for j in jitter],
                    mode = "markers",
                    name = fname,
                    marker= dict(size=4, color=[int(v*255) for v in fv_norm],
                                 colorscale="RdBu_r", opacity=0.75, showscale=False),
                    showlegend=False,
                    hovertemplate=f"<b>{fname}</b><br>SHAP: %{{x:.4f}}<extra></extra>",
                ))
            fig_bs = go.Figure(traces)
            fig_bs.update_layout(
                **PLOTLY_LAYOUT,
                title  = "SHAP Beeswarm (top 15 features)",
                xaxis_title = "SHAP value",
                yaxis  = dict(
                    tickvals=list(range(top_n)),
                    ticktext=top_feat,
                    gridcolor=PALETTE["border"],
                ),
                height = 620,
            )
            fig_bs.add_vline(x=0, line_color=PALETTE["muted"], line_width=1.5)
            st.plotly_chart(fig_bs, use_container_width=True)

    with tab_e:
        st.markdown("<div class='section-header'>Feature Group Importance</div>", unsafe_allow_html=True)
        shap_dict = {x["feature"]: x["shap_mean"] for x in shap_imp}
        group_totals = {}
        for gname, feats in FEATURE_GROUPS.items():
            group_totals[gname] = sum(shap_dict.get(f, 0) for f in feats)

        gt = pd.Series(group_totals).sort_values(ascending=True)
        colors_g = [PALETTE["accent1"] if v > gt.median() else PALETTE["accent2"] for v in gt.values]
        fig_g = go.Figure(go.Bar(
            x=gt.values, y=gt.index, orientation="h",
            marker_color=colors_g,
            text=[f"{v:.4f}" for v in gt.values], textposition="outside",
        ))
        fig_g.update_layout(**PLOTLY_LAYOUT, title="Total SHAP by Feature Group", height=400)
        st.plotly_chart(fig_g, use_container_width=True)

        # Feature detail table
        st.markdown("<div class='section-header'>Full Feature Importance Table</div>", unsafe_allow_html=True)
        if shap_imp:
            imp_df = pd.DataFrame(shap_imp).sort_values("shap_mean", ascending=False).reset_index(drop=True)
            imp_df["rank"] = range(1, len(imp_df)+1)
            imp_df = imp_df[["rank","feature","shap_mean"]]
            st.dataframe(
                imp_df.style.bar(subset=["shap_mean"], color=PALETTE["accent1"]),
                use_container_width=True,
                height=500,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — DATA SOURCES
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📡 Data Sources":
    from utils.helpers import PALETTE

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 0.3rem 0;'>
      📡 Data <span style='color:#FF6B35;'>Sources & Attribution</span>
    </h1>
    <p style='color:#8B949E; font-size:0.82rem; margin:0 0 1.5rem 0;'>
      Where and how the WildGuard AI system extracts real-time data
    </p>
    """, unsafe_allow_html=True)

    sources = [
        {
            "name": "Open-Meteo Weather API",
            "icon": "🌦️",
            "url": "https://api.open-meteo.com/v1/forecast",
            "description": "Free, open-source weather API providing real-time meteorological data. No API key required.",
            "license": "CC BY 4.0 — Open Source",
            "features": ["temperature_c", "humidity_pct", "rainfall_mm", "visibility_m (from WMO weather codes)", "wind_speed", "cloud_cover"],
            "how": "HTTP GET request with latitude/longitude parameters. Returns JSON with current weather observations. "
                   "Visibility is estimated from WMO weather codes (fog → 100-200m, clear → 900-1000m, rain → 400-700m).",
            "rate_limit": "No registration needed. 10,000 requests/day.",
            "color": "#58A6FF",
        },
        {
            "name": "OpenStreetMap Overpass API",
            "icon": "🗺️",
            "url": "https://overpass-api.de/api/interpreter",
            "description": "Queries the OpenStreetMap database for geographic features near the target location. "
                           "Returns road types, width, lighting, and water body proximity.",
            "license": "ODbL — Open Data Commons Open Database License",
            "features": ["road_type (mapped from OSM highway tag)", "road_width_m", "street_lighting",
                          "dist_water_km (Haversine distance to nearest water body)"],
            "how": "POST request with Overpass QL query. Searches within 1.5 km radius for highway=* ways and "
                   "natural=water / waterway nodes/ways. OSM highway types are mapped to model categories "
                   "(motorway→national_highway, track→forest_road, etc.).",
            "rate_limit": "Public endpoint. Max 2 requests/second, 10,000/day.",
            "color": "#06D6A0",
        },
        {
            "name": "GBIF Biodiversity API",
            "icon": "🦁",
            "url": "https://api.gbif.org/v1/occurrence/search",
            "description": "Global Biodiversity Information Facility — the world's largest open biodiversity database. "
                           "Searches for mammal occurrence records reported near the target coordinates.",
            "license": "CC BY 4.0 — GBIF.org Data Use Agreement",
            "features": ["species (dominant mammal in area)", "species_risk (mapped from species ID)",
                          "total_sightings", "biodiversity_index"],
            "how": "HTTP GET request filtered by Mammalia class (classKey=359), ±0.5° lat/lon bounding box. "
                   "Returns up to 300 occurrence records. Species names are matched against our risk mapping "
                   "(tiger=0.9, elephant=0.95, deer=0.6, etc.). Most frequent match becomes the dominant species.",
            "rate_limit": "No API key required. Rate-limited to 3 requests/second.",
            "color": "#FFD166",
        },
        {
            "name": "System Clock (Temporal Features)",
            "icon": "🕐",
            "url": "Python datetime.now()",
            "description": "Derives all temporal features from the current system time. "
                           "Indian seasonal calendar is applied (Mar-May=summer, Jun-Sep=monsoon, Oct-Nov=post_monsoon, Dec-Feb=winter).",
            "license": "N/A — Local computation",
            "features": ["hour", "day_of_week", "season", "night_flag (20:00–06:00)",
                          "dawn_dusk (05:00–07:00 / 17:00–19:00)", "rush_hour (06:00–09:00 / 17:00–20:00)",
                          "breeding_season (monsoon + post_monsoon)"],
            "how": "Python's datetime module provides the current local time. Boolean flags are computed for "
                   "nighttime, dawn/dusk (peak wildlife crossing periods), rush hour, and breeding season.",
            "rate_limit": "N/A — Instantaneous",
            "color": "#BC8CF2",
        },
        {
            "name": "GIS Computation Engine",
            "icon": "🌍",
            "url": "Haversine distance + NDVI estimation",
            "description": "Computes geographical features using a database of 21 protected areas "
                           "(14 South India + 7 Central India) and 11 wildlife corridors.",
            "license": "N/A — Local computation with openly sourced coordinates",
            "features": ["protected_dist_km (nearest tiger reserve/national park)",
                          "corridor_dist_km (nearest wildlife corridor)",
                          "ndvi (estimated from season + proximity to protected areas)",
                          "night_light (urbanization proxy from corridor distance)"],
            "how": "Haversine formula computes great-circle distance from the query point to each protected area "
                   "and corridor in our database. South India reserves include Bandipur, Nagarhole, Mudumalai, "
                   "Wayanad, Periyar, Sathyamangalam, BR Hills, Anamalai, Parambikulam, and more.",
            "rate_limit": "N/A — < 1ms computation",
            "color": "#FF6B35",
        },
        {
            "name": "News Websites (RSS Aggregator)",
            "icon": "📰",
            "url": "The Hindu | NDTV | Down to Earth | Google News RSS",
            "description": "Scrapes wildlife-vehicle collision news from major Indian news sources via RSS feeds. "
                           "Searches for recent reports of animal-road incidents, with special focus on South India "
                           "(Karnataka, Kerala, Tamil Nadu, Western Ghats). Extracts species mentions and regional tags.",
            "license": "RSS / Public Access — Editorial content",
            "features": ["total_wildlife_articles (count of matching news articles)",
                          "south_india_articles (count with South India mentions)",
                          "species_in_news (species cited in recent reports)",
                          "news_risk_modifier (risk boost from recent incidents)"],
            "how": "HTTP GET requests to RSS feed URLs for The Hindu (thehindu.com/sci-tech/energy-and-environment), "
                   "NDTV (feeds.feedburner.com/ndtv/environment-news), Down to Earth (downtoearth.org.in/rss/wildlife), "
                   "and Google News (news.google.com/rss/search?q=wildlife+animal+road+accident+India). "
                   "XML parsed with Python's xml.etree.ElementTree. Articles are filtered by wildlife keywords "
                   "(elephant, tiger, roadkill, corridor, etc.) and checked for South India state mentions.",
            "rate_limit": "No auth needed. 4 RSS feeds fetched in parallel.",
            "color": "#EF476F",
        },
        {
            "name": "Indian Government Wildlife Portals",
            "icon": "🏛️",
            "url": "NTCA | State Forest Depts | MoEFCC | WII | MoRTH | India Biodiversity Portal",
            "description": "Checks accessibility and scrapes wildlife/conservation references from official Indian "
                           "government portals. Includes the National Tiger Conservation Authority (NTCA), "
                           "state forest departments (Karnataka aranya.gov.in, Kerala forest.kerala.gov.in, "
                           "Tamil Nadu forests.tn.gov.in), Ministry of Environment (MoEFCC), Wildlife Institute of India (WII), "
                           "Ministry of Road Transport (MoRTH), and India Biodiversity Portal.",
            "license": "Government of India — Public Access",
            "features": ["govt_sources_accessible (count of reachable portals out of 6+)",
                          "conservation_intensity (tiger/corridor reference density from NTCA)",
                          "active_alerts (whether state forest dept has wildlife advisories)",
                          "nearest_state_dept (auto-detected based on location)"],
            "how": "HTTP GET requests to each government URL. HTML content is scanned with regex for wildlife, corridor, "
                   "tiger, elephant, alert, and advisory references. The nearest state forest department is auto-selected "
                   "based on the query coordinates (e.g., Bandipur → Karnataka Forest Dept at aranya.gov.in). "
                   "MoRTH is checked for road accident/safety references. India Biodiversity Portal is queried for observations.",
            "rate_limit": "Public access. 6-8 government sites checked per request.",
            "color": "#B5179E",
        },
    ]

    for src in sources:
        st.markdown(f"""
        <div class='source-card' style='border-left: 3px solid {src["color"]};'>
          <div style='display:flex; justify-content:space-between; align-items:center;'>
            <div class='source-name' style='color:{src["color"]}; font-size:1rem;'>
              {src["icon"]}  {src["name"]}
            </div>
            <span style='font-size:0.65rem; color:var(--muted); background:var(--bg);
                          padding:0.2rem 0.6rem; border-radius:4px;'>{src["license"]}</span>
          </div>
          <div style='font-size:0.8rem; color:var(--text); margin:0.5rem 0;'>{src["description"]}</div>
          <div class='source-url' style='margin:0.4rem 0;'>
            <code style='background:var(--bg); padding:0.2rem 0.5rem; border-radius:4px; font-size:0.7rem;'>
              {src["url"]}
            </code>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_feat, col_how = st.columns([1, 1.5])
        with col_feat:
            st.markdown("**Features Extracted:**")
            for feat in src["features"]:
                st.markdown(f"- `{feat}`")
        with col_how:
            st.markdown("**How It Works:**")
            st.markdown(src["how"])
            st.markdown(f"**Rate Limit:** {src['rate_limit']}")

        st.markdown("---")

    # ── Pipeline diagram ──────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔄 Pipeline Architecture (7 Sources)</div>", unsafe_allow_html=True)
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                  REAL-TIME DATA PIPELINE (v3.0)                     │
    │           7 Sources · 30+ Features · 2 Models · Ensemble           │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │   📍 User selects location (lat/lon) + speed conditions            │
    │         │                                                           │
    │  ┌──── APIs ────────────────────────────────────────────────┐      │
    │  │      ├──→ 🌦️ Open-Meteo API     ──→ weather features   │      │
    │  │      ├──→ 🗺️ Overpass API        ──→ road + water      │      │
    │  │      └──→ 🦁 GBIF API            ──→ wildlife species   │      │
    │  └──────────────────────────────────────────────────────────┘      │
    │  ┌──── News & Government ───────────────────────────────────┐      │
    │  │      ├──→ 📰 News RSS            ──→ incident intelligence│     │
    │  │      │    (The Hindu, NDTV, Down to Earth, Google News)  │      │
    │  │      └──→ 🏛️ Govt Portals        ──→ conservation data  │      │
    │  │           (NTCA, Forest Depts, MoEFCC, WII, MoRTH)      │      │
    │  └──────────────────────────────────────────────────────────┘      │
    │  ┌──── Computation ─────────────────────────────────────────┐      │
    │  │      ├──→ 🕐 System Clock        ──→ temporal features   │      │
    │  │      └──→ 🌍 GIS Engine          ──→ spatial features    │      │
    │  │           (21 PAs + 11 corridors, South + Central India) │      │
    │  └──────────────────────────────────────────────────────────┘      │
    │                   │                                                 │
    │                   ▼                                                 │
    │         ┌─────────────────┐                                        │
    │         │ Feature Vector  │  (30+ features combined)               │
    │         └────────┬────────┘                                        │
    │                  │                                                  │
    │         ┌────────┴────────┐                                        │
    │         ▼                 ▼                                         │
    │   ┌──────────┐    ┌──────────────┐                                │
    │   │ XGBoost  │    │ Random Forest│                                │
    │   │ (SHAP)   │    │ (Gini)       │                                │
    │   └────┬─────┘    └──────┬───────┘                                │
    │        └────────┬────────┘                                         │
    │                 ▼                                                   │
    │       ┌─────────────────┐                                          │
    │       │ Ensemble Result │                                          │
    │       │ + SHAP Waterfall│                                          │
    │       │ + Recommendations│                                         │
    │       └─────────────────┘                                          │
    └─────────────────────────────────────────────────────────────────────┘
    ```
    """)
