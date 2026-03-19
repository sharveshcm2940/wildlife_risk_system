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

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

:root {
  --bg:       #0D1117;
  --surface:  #161B22;
  --border:   #30363D;
  --orange:   #FF6B35;
  --teal:     #00D4AA;
  --amber:    #FFD166;
  --blue:     #58A6FF;
  --purple:   #BC8CF2;
  --low:      #06D6A0;
  --med:      #FFD166;
  --high:     #EF476F;
  --crit:     #B5179E;
  --text:     #E6EDF3;
  --muted:    #8B949E;
}

html, body, .stApp {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'Sora', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }

.metric-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.2rem 1.5rem;
  margin: 0.3rem 0;
}
.metric-card h3 {
  font-family: 'Space Mono', monospace;
  font-size: 0.72rem;
  color: var(--muted);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin: 0 0 0.4rem 0;
}
.metric-card .value {
  font-family: 'Space Mono', monospace;
  font-size: 2rem;
  font-weight: 700;
  color: var(--orange);
  line-height: 1;
}
.metric-card .sub {
  font-size: 0.78rem;
  color: var(--muted);
  margin-top: 0.3rem;
}

.risk-badge {
  display: inline-block;
  padding: 0.3rem 1rem;
  border-radius: 999px;
  font-family: 'Space Mono', monospace;
  font-weight: 700;
  font-size: 1rem;
  letter-spacing: 0.05em;
}

.section-header {
  font-family: 'Space Mono', monospace;
  font-size: 0.7rem;
  color: var(--muted);
  letter-spacing: 0.18em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.5rem;
  margin: 1.2rem 0 0.8rem 0;
}

.source-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1rem 1.2rem;
  margin: 0.5rem 0;
}
.source-card .source-name {
  font-family: 'Space Mono', monospace;
  font-size: 0.85rem;
  font-weight: 700;
  margin-bottom: 0.3rem;
}
.source-card .source-url {
  font-size: 0.7rem;
  color: var(--muted);
  word-break: break-all;
  margin-bottom: 0.4rem;
}
.source-card .source-features {
  font-size: 0.75rem;
  color: var(--teal);
}
.status-success { color: var(--teal); }
.status-fallback { color: var(--amber); }
.status-error { color: var(--high); }

[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p {
  color: var(--text) !important;
  font-size: 0.82rem !important;
}

.stSelectbox > div > div { background: var(--bg) !important; border-color: var(--border) !important; }
.stSlider [data-baseweb="slider"] { background: var(--border) !important; }
.stTabs [data-baseweb="tab-list"] { background: var(--surface); border-radius: 8px; border: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; font-family: 'Space Mono', monospace; font-size: 0.75rem; }
.stTabs [aria-selected="true"] { color: var(--orange) !important; border-bottom: 2px solid var(--orange) !important; }
.stButton > button {
  background: var(--orange) !important;
  color: white !important;
  border: none !important;
  font-family: 'Space Mono', monospace !important;
  font-weight: 700 !important;
  letter-spacing: 0.05em !important;
  border-radius: 8px !important;
  padding: 0.6rem 2rem !important;
  transition: all 0.2s !important;
}
.stButton > button:hover { background: #e55a2b !important; transform: translateY(-1px); }
div[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace !important; font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "models"

@st.cache_resource(show_spinner="🌿 Training both models on first run…")
def load_or_train():
    from models.train import WildlifeRiskModel, train_and_save
    from data.generate_data import generate_dataset

    model = WildlifeRiskModel()
    if (MODEL_DIR / "xgb_model.pkl").exists():
        model.load()
        if (MODEL_DIR / "dataset.parquet").exists():
            df = pd.read_parquet(MODEL_DIR / "dataset.parquet")
        else:
            df = generate_dataset(12_000)
    else:
        _, df = train_and_save()
        model.load()
    return model, df

model, df = load_or_train()

# Get metrics (handle both old and new format)
if "xgb" in model.metrics:
    xgb_metrics = model.metrics["xgb"]
    rf_metrics  = model.metrics.get("rf", {})
    shap_imp    = xgb_metrics.get("shap_importance", [])
    rf_imp      = rf_metrics.get("feature_importance", [])
    best_model  = model.metrics.get("best_model", "XGBoost")
else:
    # Backward compat with old single-model format
    xgb_metrics = model.metrics
    rf_metrics  = {}
    shap_imp    = model.metrics.get("shap_importance", [])
    rf_imp      = []
    best_model  = "XGBoost"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
      <div style='font-size:2.8rem;'>🐾</div>
      <div style='font-family:"Space Mono",monospace; font-size:1.1rem; color:#FF6B35; font-weight:700;'>WildGuard AI</div>
      <div style='font-size:0.7rem; color:#8B949E; letter-spacing:0.12em;'>WILDLIFE RISK SYSTEM v3.0</div>
      <div style='font-size:0.65rem; color:#58A6FF; letter-spacing:0.08em; margin-top:0.3rem;'>REAL-TIME DATA PIPELINE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 Live Risk Predictor", "🗺 Risk Map",
         "📊 Analytics", "🧠 Model Insights", "📡 Data Sources"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    xgb_auc = xgb_metrics.get('roc_auc', 0)
    rf_auc  = rf_metrics.get('roc_auc', 0)
    st.markdown(f"""
    <div class='metric-card' style='margin-top:0.5rem;'>
      <h3>XGBoost</h3>
      <div class='value' style='font-size:0.9rem; color:#FF6B35;'>● ROC-AUC: {xgb_auc:.4f}</div>
    </div>
    <div class='metric-card' style='margin-top:0.3rem;'>
      <h3>Random Forest</h3>
      <div class='value' style='font-size:0.9rem; color:#58A6FF;'>● ROC-AUC: {rf_auc:.4f}</div>
    </div>
    <div class='metric-card' style='margin-top:0.3rem;'>
      <h3>Best Model</h3>
      <div class='value' style='font-size:0.85rem; color:#00D4AA;'>✦ {best_model}</div>
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

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:2rem; margin:0 0 0.2rem 0;'>
      Wildlife-Vehicle Collision
      <span style='color:#FF6B35;'>Risk Intelligence</span>
    </h1>
    <p style='color:#8B949E; font-size:0.88rem; margin:0 0 1.5rem 0;'>
      Dual-model ML system (XGBoost + Random Forest) with real-time data pipeline
    </p>
    """, unsafe_allow_html=True)

    # ── KPI Row ────────────────────────────────────────────────────────────────
    acc_rate  = df["accident"].mean()
    high_risk = (df["risk_score"] > 0.65).sum()
    top_sp    = df[df["accident"]==1]["species"].value_counts().idxmax()
    top_road  = df[df["accident"]==1]["road_type"].value_counts().idxmax()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (c1, "Total Records",    f"{len(df):,}",                     "Training data"),
        (c2, "Accident Rate",    f"{acc_rate:.1%}",                  "Overall"),
        (c3, "High-Risk Zones",  f"{high_risk:,}",                   "Risk > 65%"),
        (c4, "Riskiest Species", top_sp.capitalize(),                 "Most accidents"),
        (c5, "XGB AUC",          f"{xgb_metrics.get('roc_auc',0):.4f}", "XGBoost"),
        (c6, "RF AUC",           f"{rf_metrics.get('roc_auc',0):.4f}",  "Random Forest"),
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
    from data.realtime_extractor import RealtimeDataExtractor

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 0.3rem 0;'>
      🔮 Live <span style='color:#FF6B35;'>Risk Prediction</span> Pipeline
    </h1>
    <p style='color:#8B949E; font-size:0.82rem; margin:0 0 1.2rem 0;'>
      Extracts real-time data from APIs, news sites & govt portals → feeds into both models → shows results
    </p>
    """, unsafe_allow_html=True)

    # ── Location & speed inputs ───────────────────────────────────────────────
    st.markdown("<div class='section-header'>📍 Select Location & Conditions</div>", unsafe_allow_html=True)

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
        lat = st.number_input("Latitude",  min_value=8.0, max_value=25.0, value=default_lat, step=0.01)
    with c3:
        lon = st.number_input("Longitude", min_value=74.0, max_value=82.0, value=default_lon, step=0.01)

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
# PAGE 3 — RISK MAP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗺 Risk Map":
    from streamlit_folium import st_folium
    from utils.helpers import build_folium_map

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 0.5rem 0;'>
      🗺 Geospatial <span style='color:#FF6B35;'>Risk Map</span>
    </h1>
    <p style='color:#8B949E; font-size:0.85rem; margin:0 0 1rem 0;'>
      Kernel density heatmap of historical wildlife-vehicle collisions
    </p>
    """, unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1: f_season  = st.multiselect("Filter Season", df["season"].unique().tolist(), default=df["season"].unique().tolist())
    with fc2: f_species = st.multiselect("Filter Species", df["species"].unique().tolist(), default=df["species"].unique().tolist())
    with fc3: f_road    = st.multiselect("Filter Road Type", df["road_type"].unique().tolist(), default=df["road_type"].unique().tolist())

    map_df = df[
        df["season"].isin(f_season) &
        df["species"].isin(f_species) &
        df["road_type"].isin(f_road)
    ]

    c_left, c_right = st.columns([3, 1])
    with c_left:
        m = build_folium_map(map_df)
        st_folium(m, width=None, height=600, returned_objects=[])

    with c_right:
        st.markdown("<div class='section-header'>Map Legend</div>", unsafe_allow_html=True)
        for color, label in [("#06D6A0","Low Risk"),("#FFD166","Moderate"),("#EF476F","High Risk"),("#B5179E","Critical")]:
            st.markdown(f"<div style='display:flex;align-items:center;gap:0.5rem;margin:0.3rem 0;'><div style='width:14px;height:14px;border-radius:50%;background:{color};'></div><span style='font-size:0.82rem;'>{label}</span></div>", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Summary</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='metric-card'>
          <h3>Records shown</h3><div class='value' style='font-size:1.4rem;'>{len(map_df):,}</div>
        </div>
        <div class='metric-card'>
          <h3>Accident events</h3><div class='value' style='font-size:1.4rem;'>{map_df["accident"].sum():,}</div>
        </div>
        <div class='metric-card'>
          <h3>Avg risk score</h3><div class='value' style='font-size:1.4rem;'>{map_df["risk_score"].mean():.3f}</div>
        </div>
        """, unsafe_allow_html=True)


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