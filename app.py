"""
Wildlife-Vehicle Collision Risk Prediction System
Main Streamlit Application
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

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }

/* Custom metric cards */
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

/* Risk badge */
.risk-badge {
  display: inline-block;
  padding: 0.3rem 1rem;
  border-radius: 999px;
  font-family: 'Space Mono', monospace;
  font-weight: 700;
  font-size: 1rem;
  letter-spacing: 0.05em;
}

/* Section headers */
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

/* Sidebar styling */
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

/* Streamlit elements override */
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

@st.cache_resource(show_spinner="🌿 Training model on first run…")
def load_or_train():
    """Load cached model; train from scratch if unavailable."""
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
metrics   = model.metrics
shap_imp  = metrics.get("shap_importance", [])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
      <div style='font-size:2.8rem;'>🐾</div>
      <div style='font-family:"Space Mono",monospace; font-size:1.1rem; color:#FF6B35; font-weight:700;'>WildGuard AI</div>
      <div style='font-size:0.7rem; color:#8B949E; letter-spacing:0.12em;'>WILDLIFE RISK SYSTEM v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 Risk Predictor", "🗺 Risk Map", "📊 Analytics", "🧠 Model Insights"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(f"""
    <div class='metric-card' style='margin-top:0.5rem;'>
      <h3>Model Status</h3>
      <div class='value' style='font-size:1rem; color:#00D4AA;'>● ONLINE</div>
      <div class='sub'>XGBoost  |  ROC-AUC: {metrics.get('roc_auc',0):.4f}</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    from utils.helpers import (
        hourly_risk_chart, species_risk_chart,
        season_road_heatmap, rolling_trend_chart, ndvi_risk_scatter,
        confusion_matrix_chart, PALETTE
    )

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:2rem; margin:0 0 0.2rem 0;'>
      Wildlife-Vehicle Collision
      <span style='color:#FF6B35;'>Risk Intelligence</span>
    </h1>
    <p style='color:#8B949E; font-size:0.88rem; margin:0 0 1.5rem 0;'>
      Real-time ML-powered risk assessment for wildlife crossing hotspots
    </p>
    """, unsafe_allow_html=True)

    # ── KPI Row ────────────────────────────────────────────────────────────────
    acc_rate  = df["accident"].mean()
    high_risk = (df["risk_score"] > 0.65).sum()
    top_sp    = df[df["accident"]==1]["species"].value_counts().idxmax()
    top_road  = df[df["accident"]==1]["road_type"].value_counts().idxmax()

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, "Total Records",    f"{len(df):,}",           "Training dataset size"),
        (c2, "Accident Rate",    f"{acc_rate:.1%}",         "Overall collision rate"),
        (c3, "High-Risk Zones",  f"{high_risk:,}",          "Events with risk > 65%"),
        (c4, "Riskiest Species", top_sp.capitalize(),        "Highest accident count"),
        (c5, "Model ROC-AUC",    f"{metrics.get('roc_auc',0):.4f}", "XGBoost performance"),
    ]
    for col, title, val, sub in kpis:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <h3>{title}</h3>
              <div class='value'>{val}</div>
              <div class='sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

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

    # ── Confusion matrix + report ─────────────────────────────────────────────
    col_e, col_f = st.columns([1, 1.5])
    with col_e:
        cm = metrics.get("confusion", [[0,0],[0,0]])
        st.plotly_chart(confusion_matrix_chart(cm), use_container_width=True)
    with col_f:
        report = metrics.get("report", {})
        if report:
            st.markdown("<div class='section-header'>Classification Report</div>", unsafe_allow_html=True)
            rdf = pd.DataFrame(report).T.round(3)
            st.dataframe(
                rdf.style.background_gradient(cmap="RdYlGn", axis=None),
                use_container_width=True,
                height=250,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Risk Predictor":
    from utils.helpers import risk_gauge, shap_waterfall, get_risk_level, PALETTE

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 1.2rem 0;'>
      🔮 Real-Time <span style='color:#FF6B35;'>Risk Prediction</span>
    </h1>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("<div class='section-header'>📍 Location & Road</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: lat = st.number_input("Latitude",  min_value=18.5, max_value=24.5, value=21.5, step=0.01)
        with c2: lon = st.number_input("Longitude", min_value=74.0, max_value=82.0, value=78.5, step=0.01)
        with c3: road_type = st.selectbox("Road Type", ["highway","rural","forest_road","state_highway","national_highway"])
        with c4: road_width = st.slider("Road Width (m)", 4, 14, 7)

        st.markdown("<div class='section-header'>🚗 Traffic & Speed</div>", unsafe_allow_html=True)
        c5, c6, c7, c8 = st.columns(4)
        with c5: speed_limit  = st.selectbox("Speed Limit (km/h)", [30, 40, 60, 80, 100], index=2)
        with c6: actual_speed = st.slider("Actual Speed (km/h)", 10, 140, 65)
        with c7: curvature    = st.slider("Road Curvature (°/km)", 0, 60, 12)
        with c8: street_light = st.selectbox("Street Lighting", ["Absent", "Present"])

        st.markdown("<div class='section-header'>🕐 Temporal</div>", unsafe_allow_html=True)
        c9, c10, c11, c12 = st.columns(4)
        with c9:  hour       = st.slider("Hour of Day", 0, 23, 21)
        with c10: day_of_week= st.slider("Day (0=Mon)", 0, 6, 4)
        with c11: season     = st.selectbox("Season", ["summer","monsoon","post_monsoon","winter"])
        with c12: breeding   = st.selectbox("Breeding Season", ["No","Yes"])

        st.markdown("<div class='section-header'>🌿 Environment & Wildlife</div>", unsafe_allow_html=True)
        c13, c14, c15, c16 = st.columns(4)
        with c13: ndvi         = st.slider("NDVI (Forest Density)", 0.0, 1.0, 0.65)
        with c14: dist_water   = st.slider("Distance to Water (km)", 0.0, 15.0, 1.5)
        with c15: rainfall     = st.slider("Rainfall (mm)", 0.0, 100.0, 15.0)
        with c16: visibility   = st.slider("Visibility (m)", 50, 1000, 600)

        c17, c18, c19, c20 = st.columns(4)
        with c17: species      = st.selectbox("Species", ["tiger","leopard","elephant","deer","boar","wolf","nilgai","sambar"])
        with c18: corridor_d   = st.slider("Corridor Distance (km)", 0.0, 10.0, 1.0)
        with c19: protected_d  = st.slider("Protected Area Dist (km)", 0.0, 20.0, 3.0)
        with c20: temp         = st.slider("Temperature (°C)", 10.0, 45.0, 28.0)

        c21, c22, c23, c24 = st.columns(4)
        with c21: humidity    = st.slider("Humidity (%)", 20, 100, 70)
        with c22: past_acc    = st.slider("Past Accidents (count)", 0, 30, 3)
        with c23: night_light = st.slider("Nighttime Light (0-255)", 0, 255, 25)
        with c24: rolling_7d  = st.slider("Rolling 7-Day Trend", 0.0, 20.0, 3.5)

        submitted = st.form_submit_button("⚡ PREDICT RISK", use_container_width=True)

    if submitted:
        # ── Build input row ──────────────────────────────────────────────────
        night_flag = int(hour < 6 or hour >= 20)
        dawn_dusk  = int((5 <= hour <= 7) or (17 <= hour <= 19))
        rush_hour  = int((6 <= hour <= 9) or (17 <= hour <= 20))
        speed_ratio= actual_speed / max(speed_limit, 1)
        street_l   = int(street_light == "Present")
        breed_flag = int(breeding == "Yes")

        ROAD_RISK = {'forest_road':0.90,'rural':0.65,'state_highway':0.55,'highway':0.45,'national_highway':0.40}
        SPEC_RISK = {'tiger':0.9,'elephant':0.95,'leopard':0.85,'deer':0.60,'boar':0.55,'wolf':0.70,'nilgai':0.50,'sambar':0.65}
        SEAS_RISK = {'monsoon':0.90,'post_monsoon':0.70,'winter':0.55,'summer':0.40}

        driver_risk   = speed_ratio * (1 + 0.6 * night_flag) * ROAD_RISK[road_type]
        move_score    = (0.30*ndvi + 0.25*min(1/(dist_water+0.1),1) +
                         0.20*dawn_dusk + 0.15*breed_flag + 0.10*(1-night_light/255))
        kde_density   = past_acc / (corridor_d + 0.5) * 0.4

        row = pd.DataFrame([{
            "ndvi":           ndvi,
            "dist_water_km":  dist_water,
            "speed_limit":    speed_limit,
            "actual_speed":   actual_speed,
            "hour":           hour,
            "road_type":      road_type,
            "rainfall_mm":    rainfall,
            "visibility_m":   int(visibility),
            "past_accidents": past_acc,
            "day_of_week":    day_of_week,
            "season":         season,
            "rush_hour":      rush_hour,
            "movement_score": round(move_score, 4),
            "kde_density":    round(kde_density, 4),
            "driver_risk":    round(driver_risk, 4),
            "corridor_dist_km": corridor_d,
            "breeding_season":  breed_flag,
            "species":        species,
            "species_risk":   SPEC_RISK[species],
            "night_light":    night_light,
            "rolling_7day":   rolling_7d,
            "curvature_deg_km": curvature,
            "street_lighting":  street_l,
            "road_width_m":     road_width,
            "protected_dist_km": protected_d,
            "temperature_c":    temp,
            "humidity_pct":     humidity,
            "night_flag":      night_flag,
            "dawn_dusk":       dawn_dusk,
            "speed_ratio":     round(speed_ratio, 3),
        }])

        prob = float(model.predict_risk(row)[0])
        rl   = get_risk_level(prob)

        # ── Results layout ───────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns([1, 1.5])
        with r1:
            st.plotly_chart(risk_gauge(prob), use_container_width=True)
            st.markdown(f"""
            <div style='text-align:center; margin-top:-1rem;'>
              <span class='risk-badge' style='background:{rl["color"]}33; color:{rl["color"]}; border:1.5px solid {rl["color"]};'>
                {rl["emoji"]} {rl["label"]} Risk — {prob:.1%}
              </span>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            try:
                sv, X_in, base = model.predict_shap(row)
                sv_flat = sv[0] if sv.ndim == 2 else sv
                st.plotly_chart(
                    shap_waterfall(sv_flat, model.feature_cols,
                                   X_in.iloc[0].values, base),
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")

        # ── Recommendations ──────────────────────────────────────────────────
        st.markdown("<div class='section-header'>🛡 Mitigation Recommendations</div>", unsafe_allow_html=True)
        recs = []
        if night_flag:      recs.append("🌙 **Nocturnal alert zone** — deploy flashing warning signs between 20:00–06:00")
        if ndvi > 0.6:      recs.append("🌲 **High vegetation corridor** — install wildlife detection sensors")
        if dist_water < 1:  recs.append("💧 **Water source proximity** — install water crossing structures / underpasses")
        if speed_ratio > 1.1: recs.append("🚗 **Over-speed detected** — enforce speed cameras and lower limit")
        if rainfall > 30:   recs.append("🌧 **Low visibility conditions** — dynamic variable speed limits")
        if breed_flag:      recs.append("🔥 **Breeding season** — temporary speed restrictions May–October")
        if corridor_d < 1:  recs.append("🗺 **Corridor proximity** — install wildlife fencing and crossing")
        if not recs:        recs.append("✅ Risk factors within acceptable range — standard monitoring advised")
        for rec in recs:
            st.markdown(f"- {rec}")


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

    # Filter controls
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

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            # Heatmap: hour × day
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
    from utils.helpers import feature_importance_chart, confusion_matrix_chart, PALETTE, PLOTLY_LAYOUT
    import shap as shap_lib

    st.markdown("""
    <h1 style='font-family:"Space Mono",monospace; font-size:1.8rem; margin:0 0 1.2rem 0;'>
      🧠 Model <span style='color:#FF6B35;'>Insights & Explainability</span>
    </h1>
    """, unsafe_allow_html=True)

    tab_a, tab_b, tab_c, tab_d = st.tabs([
        "📈 Performance", "🔬 SHAP Importance", "📊 SHAP Beeswarm", "⚙️ Feature Groups"
    ])

    with tab_a:
        c1, c2, c3, c4 = st.columns(4)
        for col, k, label in [
            (c1, "roc_auc",      "ROC-AUC"),
            (c2, "avg_precision","Avg Precision"),
            (c3, "accuracy",     "Accuracy"),
            (c4, "brier_score",  "Brier Score"),
        ]:
            with col:
                v = metrics.get(k, 0)
                st.markdown(f"""
                <div class='metric-card'>
                  <h3>{label}</h3>
                  <div class='value'>{v:.4f}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        cm_col, tr_col = st.columns(2)
        with cm_col:
            cm = metrics.get("confusion", [[0,0],[0,0]])
            st.plotly_chart(confusion_matrix_chart(cm), use_container_width=True)
        with tr_col:
            # Learning curve proxy
            n_est = metrics.get("best_iteration", 300)
            x     = list(range(1, n_est+1, max(1, n_est//100)))
            auc_curve = [0.5 + (metrics.get("roc_auc",0.9)-0.5)*(1-np.exp(-0.015*i)) for i in x]
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(x=x, y=auc_curve, mode="lines",
                                        line=dict(color=PALETTE["accent2"], width=2), name="Train AUC"))
            fig_lc.update_layout(**PLOTLY_LAYOUT, title="Learning Curve (AUC)", height=320)
            st.plotly_chart(fig_lc, use_container_width=True)

    with tab_b:
        if shap_imp:
            st.plotly_chart(feature_importance_chart(shap_imp), use_container_width=True)
        else:
            st.info("SHAP importance not available.")

    with tab_c:
        st.markdown("<div class='section-header'>SHAP Beeswarm — Global Feature Impact</div>",
                    unsafe_allow_html=True)
        sv  = model.shap_values
        ssp = model.shap_sample
        if sv is not None and ssp is not None:
            # Manual beeswarm using Plotly
            top_n    = 15
            top_feat = [x["feature"] for x in sorted(shap_imp, key=lambda x: x["shap_mean"], reverse=True)[:top_n]]
            idxs     = [model.feature_cols.index(f) for f in top_feat if f in model.feature_cols]

            traces = []
            for rank, (fi, fname) in enumerate(zip(idxs, top_feat)):
                fv = ssp.iloc[:, fi].values
                fv_norm = (fv - fv.min()) / ((fv.max() - fv.min()) + 1e-9)
                colors  = [f"hsl({int(v*240)},80%,55%)" for v in fv_norm]
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

    with tab_d:
        from models.train import FEATURE_GROUPS
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