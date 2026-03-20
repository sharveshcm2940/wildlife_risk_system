"""
Utility functions — Charts, maps, risk formatters, and model comparison helpers
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional


# ── Colour palette — Palantir Foundry Aesthetic ──────────────────────────────
PALETTE = {
    "bg":       "#060a10",
    "surface":  "#0d1320",
    "border":   "#1a2332",
    "accent1":  "#00e5ff",
    "accent2":  "#00e676",
    "accent3":  "#ffb020",
    "low":      "#00e676",
    "medium":   "#ffb020",
    "high":     "#ff3d5a",
    "critical": "#d500f9",
    "text":     "#c8d6e5",
    "muted":    "#5a6d82",
    "blue":     "#4da6ff",
    "purple":   "#9d7aff",
    "orange":   "#ff7043",
    "cyan":     "#00e5ff",
}

RISK_LEVELS = [
    (0.0,  0.25, "Low",      PALETTE["low"],      "🟢"),
    (0.25, 0.50, "Moderate", PALETTE["medium"],   "🟡"),
    (0.50, 0.75, "High",     PALETTE["high"],     "🔴"),
    (0.75, 1.00, "Critical", PALETTE["critical"], "🟣"),
]

PLOTLY_LAYOUT = dict(
    paper_bgcolor = PALETTE["bg"],
    plot_bgcolor  = PALETTE["surface"],
    font          = dict(color=PALETTE["text"], family="'JetBrains Mono', 'IBM Plex Mono', monospace"),
    margin        = dict(l=40, r=20, t=50, b=40),
)

_AX = dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"])


def get_risk_level(prob: float) -> dict:
    for lo, hi, label, color, emoji in RISK_LEVELS:
        if lo <= prob < hi or (prob >= 0.75 and hi == 1.00):
            return {"label": label, "color": color, "emoji": emoji, "prob": prob}
    return {"label": "Critical", "color": PALETTE["critical"], "emoji": "🟣", "prob": prob}


# ── Folium map ────────────────────────────────────────────────────────────────
def build_folium_map(df: pd.DataFrame, center: list = [21.5, 78.5],
                     zoom: int = 6) -> folium.Map:
    m = folium.Map(
        location      = center,
        zoom_start    = zoom,
        tiles         = "CartoDB dark_matter",
        control_scale = True,
    )
    acc = df[df["accident"] == 1][["latitude", "longitude", "risk_score"]].dropna()
    heat_data = [[r.latitude, r.longitude, r.risk_score] for _, r in acc.iterrows()]
    HeatMap(
        heat_data,
        min_opacity = 0.3,
        max_zoom    = 14,
        radius      = 18,
        blur        = 15,
        gradient    = {"0.2": "#06D6A0", "0.5": "#FFD166",
                       "0.75": "#EF476F", "1.0": "#B5179E"},
    ).add_to(m)

    high    = df[df["risk_score"] > 0.70].head(200)
    cluster = MarkerCluster(name="High Risk Zones").add_to(m)
    for _, row in high.iterrows():
        rl = get_risk_level(row["risk_score"])
        folium.CircleMarker(
            location     = [row["latitude"], row["longitude"]],
            radius       = 7,
            color        = rl["color"],
            fill         = True,
            fill_opacity = 0.85,
            popup        = folium.Popup(
                f"<b>{rl['emoji']} {rl['label']}</b><br>"
                f"Risk: {row['risk_score']:.2%}<br>"
                f"Species: {row.get('species','–')}<br>"
                f"Road: {row.get('road_type','–')}<br>"
                f"Hour: {int(row.get('hour',0)):02d}:00",
                max_width=200,
            ),
        ).add_to(cluster)

    folium.LayerControl().add_to(m)
    return m


# ── Chart functions ───────────────────────────────────────────────────────────

def risk_gauge(prob: float) -> go.Figure:
    rl  = get_risk_level(prob)
    fig = go.Figure(go.Indicator(
        mode   = "gauge+number+delta",
        value  = prob * 100,
        number = {"suffix": "%", "font": {"size": 44, "color": rl["color"]}},
        delta  = {"reference": 50, "valueformat": ".1f"},
        gauge  = {
            "axis":       {"range": [0, 100], "tickfont": {"color": PALETTE["text"]}},
            "bar":        {"color": rl["color"]},
            "bgcolor":    PALETTE["surface"],
            "bordercolor": PALETTE["border"],
            "steps": [
                {"range": [0,  25], "color": "#0a2e20"},
                {"range": [25, 50], "color": "#2e2810"},
                {"range": [50, 75], "color": "#2e1010"},
                {"range": [75,100], "color": "#1a0a1e"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "value": prob * 100},
        },
        title = {"text": f"{rl['emoji']} {rl['label']} Risk",
                 "font": {"color": rl["color"], "size": 18}},
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=280)
    return fig


def dual_risk_gauge(xgb_prob: float, rf_prob: float) -> go.Figure:
    """Side-by-side gauges for both models."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=["XGBoost", "Random Forest"],
    )

    for col, (prob, name) in enumerate([(xgb_prob, "XGBoost"), (rf_prob, "RF")], 1):
        rl = get_risk_level(prob)
        fig.add_trace(go.Indicator(
            mode   = "gauge+number",
            value  = prob * 100,
            number = {"suffix": "%", "font": {"size": 36, "color": rl["color"]}},
            gauge  = {
                "axis":       {"range": [0, 100], "tickfont": {"color": PALETTE["text"]}},
                "bar":        {"color": rl["color"]},
                "bgcolor":    PALETTE["surface"],
                "bordercolor": PALETTE["border"],
                "steps": [
                    {"range": [0,  25], "color": "#0a2e20"},
                    {"range": [25, 50], "color": "#2e2810"},
                    {"range": [50, 75], "color": "#2e1010"},
                    {"range": [75,100], "color": "#1a0a1e"},
                ],
            },
            title = {"text": f"{rl['emoji']} {name}",
                     "font": {"color": rl["color"], "size": 14}},
        ), row=1, col=col)

    fig.update_layout(**PLOTLY_LAYOUT, height=280)
    return fig


def feature_importance_chart(shap_importance: list) -> go.Figure:
    top  = sorted(shap_importance, key=lambda x: x["shap_mean"], reverse=True)[:20]
    feat = [x["feature"] for x in top][::-1]
    vals = [x["shap_mean"] for x in top][::-1]
    colors = [PALETTE["accent1"] if v > np.median(vals) else PALETTE["accent2"] for v in vals]

    fig = go.Figure(go.Bar(
        x           = vals,
        y           = feat,
        orientation = "h",
        marker      = dict(color=colors, line=dict(width=0)),
        hovertemplate = "<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title  = "Top Feature Importances (mean |SHAP|)",
        xaxis  = dict(title="Mean |SHAP value|", **_AX),
        yaxis  = dict(**_AX),
        height = 600,
    )
    return fig


def rf_importance_chart(rf_importance: list) -> go.Figure:
    top  = sorted(rf_importance, key=lambda x: x["importance"], reverse=True)[:20]
    feat = [x["feature"] for x in top][::-1]
    vals = [x["importance"] for x in top][::-1]
    colors = [PALETTE["blue"] if v > np.median(vals) else PALETTE["purple"] for v in vals]

    fig = go.Figure(go.Bar(
        x           = vals,
        y           = feat,
        orientation = "h",
        marker      = dict(color=colors, line=dict(width=0)),
        hovertemplate = "<b>%{y}</b><br>Gini: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title  = "Random Forest — Gini Feature Importance",
        xaxis  = dict(title="Gini Importance", **_AX),
        yaxis  = dict(**_AX),
        height = 600,
    )
    return fig


def model_comparison_chart(xgb_m: dict, rf_m: dict) -> go.Figure:
    metrics_list = ["roc_auc", "avg_precision", "accuracy", "brier_score"]
    labels       = ["ROC-AUC", "Avg Precision", "Accuracy", "Brier Score"]
    xgb_vals     = [xgb_m.get(m, 0) for m in metrics_list]
    rf_vals      = [rf_m.get(m, 0) for m in metrics_list]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=xgb_vals, name="XGBoost",
        marker_color=PALETTE["accent1"],
        text=[f"{v:.4f}" for v in xgb_vals], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=labels, y=rf_vals, name="Random Forest",
        marker_color=PALETTE["blue"],
        text=[f"{v:.4f}" for v in rf_vals], textposition="outside",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title    = "Model Performance Comparison",
        barmode  = "group",
        xaxis    = dict(**_AX),
        yaxis    = dict(**_AX),
        legend   = dict(x=0.7, y=0.99, bgcolor=PALETTE["surface"]),
        height   = 380,
    )
    return fig


def roc_comparison_chart(xgb_m: dict, rf_m: dict) -> go.Figure:
    fig = go.Figure()
    # XGBoost ROC
    xgb_roc = xgb_m.get("roc_curve", {})
    if xgb_roc:
        fig.add_trace(go.Scatter(
            x=xgb_roc["fpr"], y=xgb_roc["tpr"],
            mode="lines", name=f"XGBoost (AUC={xgb_m.get('roc_auc',0):.4f})",
            line=dict(color=PALETTE["accent1"], width=2.5),
        ))
    # RF ROC
    rf_roc = rf_m.get("roc_curve", {})
    if rf_roc:
        fig.add_trace(go.Scatter(
            x=rf_roc["fpr"], y=rf_roc["tpr"],
            mode="lines", name=f"RF (AUC={rf_m.get('roc_auc',0):.4f})",
            line=dict(color=PALETTE["blue"], width=2.5),
        ))
    # Diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color=PALETTE["muted"], dash="dash", width=1),
        showlegend=False,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title  = "ROC Curve Comparison",
        xaxis  = dict(title="False Positive Rate", **_AX),
        yaxis  = dict(title="True Positive Rate", **_AX),
        legend = dict(x=0.55, y=0.05, bgcolor=PALETTE["surface"]),
        height = 380,
    )
    return fig


def hourly_risk_chart(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("hour")["accident"].agg(["mean", "count"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grp["hour"], y=grp["count"], name="Event Count",
        marker_color=PALETTE["accent2"], opacity=0.5, yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=grp["hour"], y=grp["mean"], mode="lines+markers",
        name="Accident Rate",
        line=dict(color=PALETTE["accent1"], width=2.5),
        marker=dict(size=8),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title  = "Accident Rate & Volume by Hour of Day",
        xaxis  = dict(title="Hour", **_AX),
        yaxis  = dict(title="Accident Rate", **_AX),
        yaxis2 = dict(title="Count", overlaying="y", side="right", **_AX),
        legend = dict(x=0.01, y=0.99, bgcolor=PALETTE["surface"]),
        height = 360,
    )
    return fig


def species_risk_chart(df: pd.DataFrame) -> go.Figure:
    grp    = (df.groupby("species")["accident"]
                .agg(["mean", "count"])
                .reset_index()
                .sort_values("mean", ascending=True))
    colors = [PALETTE["critical"] if v > 0.6 else PALETTE["high"] if v > 0.4 else PALETTE["medium"]
              for v in grp["mean"]]
    fig = go.Figure(go.Bar(
        x=grp["mean"], y=grp["species"], orientation="h",
        marker_color=colors,
        text=[f"{v:.1%}" for v in grp["mean"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Rate: %{x:.1%}<br>Count: %{customdata}<extra></extra>",
        customdata=grp["count"],
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title  = "Accident Rate by Species",
        xaxis  = dict(title="Accident Rate", tickformat=".0%", **_AX),
        yaxis  = dict(**_AX),
        height = 360,
    )
    return fig


def season_road_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot_table(index="season", columns="road_type",
                           values="accident", aggfunc="mean")
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
        colorscale=[[0, PALETTE["low"]], [0.5, PALETTE["medium"]], [1, PALETTE["critical"]]],
        text=[[f"{v:.1%}" for v in row] for row in pivot.values],
        texttemplate="%{text}", showscale=True, hoverongaps=False,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title  = "Accident Rate: Season × Road Type",
        xaxis  = dict(**_AX),
        yaxis  = dict(**_AX),
        height = 360,
    )
    return fig


def shap_waterfall(shap_vals, feature_names, feature_values, base_value):
    pairs = sorted(zip(shap_vals, feature_names, feature_values),
                   key=lambda x: abs(x[0]), reverse=True)[:15]
    sv, fn, fv = zip(*pairs)
    sv     = list(sv)[::-1]
    fn     = [f"{n}={v:.2f}" if isinstance(v, float) else f"{n}={v}"
              for n, v in zip(fn, fv)][::-1]
    colors = [PALETTE["high"] if s > 0 else PALETTE["accent2"] for s in sv]

    fig = go.Figure(go.Bar(
        x=sv, y=fn, orientation="h", marker_color=colors,
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=PALETTE["muted"], line_width=1.5)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title  = f"SHAP Explanation  (base = {base_value:.3f})",
        xaxis  = dict(title="SHAP value  (impact on log-odds)", **_AX),
        yaxis  = dict(**_AX),
        height = 520,
    )
    return fig


def rolling_trend_chart(df: pd.DataFrame) -> go.Figure:
    roll = df["accident"].rolling(200, min_periods=50).mean().reset_index(drop=True)
    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        y=roll, mode="lines",
        line=dict(color=PALETTE["accent1"], width=2),
        name="Rolling accident rate",
        fill="tozeroy", fillcolor="rgba(255,107,53,0.15)",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title  = "Rolling Accident Rate Trend (200-event window)",
        xaxis  = dict(title="Record Index", **_AX),
        yaxis  = dict(title="Accident Rate", **_AX),
        height = 280,
    )
    return fig


def ndvi_risk_scatter(df: pd.DataFrame) -> go.Figure:
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="ndvi", y="risk_score", color="species",
        size="past_accidents", size_max=14, opacity=0.65,
        title="NDVI vs Risk Score by Species",
        labels={"ndvi": "Forest Density (NDVI)", "risk_score": "Risk Score"},
    )
    fig.update_layout(**PLOTLY_LAYOUT, xaxis=dict(**_AX), yaxis=dict(**_AX), height=420)
    return fig


def confusion_matrix_chart(cm: list, title: str = "Confusion Matrix") -> go.Figure:
    labels = ["No Accident", "Accident"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, PALETTE["surface"]], [1, PALETTE["accent1"]]],
        text=cm, texttemplate="%{text}", showscale=False,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title  = title,
        xaxis  = dict(title="Predicted", **_AX),
        yaxis  = dict(title="Actual", **_AX),
        height = 300,
    )
    return fig


def data_source_status_chart(log: list) -> go.Figure:
    """Horizontal bar chart showing API response times and status."""
    sources  = [l["source"] for l in log][::-1]
    times    = [l["response_ms"] for l in log][::-1]
    statuses = [l["status"] for l in log][::-1]

    colors = []
    for s in statuses:
        if s == "success":
            colors.append(PALETTE["accent2"])
        elif s == "fallback":
            colors.append(PALETTE["accent3"])
        else:
            colors.append(PALETTE["high"])

    fig = go.Figure(go.Bar(
        x=times, y=sources, orientation="h",
        marker_color=colors,
        text=[f"{t}ms — {s}" for t, s in zip(times, statuses)],
        textposition="outside",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title  = "Data Source Response Times",
        xaxis  = dict(title="Response Time (ms)", **_AX),
        yaxis  = dict(**_AX),
        height = 300,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ANIMAL MOVEMENT CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def movement_score_by_species(df: pd.DataFrame) -> go.Figure:
    """Radar chart showing movement score components by species."""
    species_list = df["species"].unique().tolist()
    metrics = ["movement_score", "dawn_dusk", "breeding_season", "night_flag"]
    metric_labels = ["Movement Score", "Dawn/Dusk Activity", "Breeding Season", "Night Activity"]

    fig = go.Figure()
    colors = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"],
              PALETTE["high"], PALETTE["critical"], PALETTE["blue"],
              PALETTE["purple"], PALETTE["orange"]]

    for i, sp in enumerate(species_list):
        sp_data = df[df["species"] == sp]
        vals = [sp_data[m].mean() for m in metrics]
        vals.append(vals[0])
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=metric_labels + [metric_labels[0]],
            fill="toself",
            name=sp.capitalize(),
            line=dict(color=colors[i % len(colors)], width=2),
            opacity=0.7,
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Species Movement Profile (Radar)",
        polar=dict(
            bgcolor=PALETTE["surface"],
            radialaxis=dict(gridcolor=PALETTE["border"], color=PALETTE["text"]),
            angularaxis=dict(gridcolor=PALETTE["border"], color=PALETTE["text"]),
        ),
        legend=dict(x=1.05, y=1, bgcolor=PALETTE["surface"]),
        height=480,
    )
    return fig


def movement_hourly_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of movement score by hour and species."""
    pivot = df.pivot_table(index="species", columns="hour",
                           values="movement_score", aggfunc="mean")
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=[s.capitalize() for s in pivot.index],
        colorscale=[
            [0, "#0a1628"], [0.3, "#06D6A0"],
            [0.6, "#FFD166"], [1.0, "#EF476F"]
        ],
        showscale=True,
        hoverongaps=False,
        hovertemplate="<b>%{y}</b> at %{x}<br>Movement: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Animal Movement Intensity by Hour & Species",
        xaxis=dict(title="Hour of Day", **_AX),
        yaxis=dict(**_AX),
        height=400,
    )
    return fig


def movement_corridor_chart(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of corridor distance vs movement score by species."""
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="corridor_dist_km", y="movement_score",
        color="species", size="risk_score", size_max=12,
        opacity=0.6,
        title="Wildlife Corridor Proximity vs Movement Score",
        labels={
            "corridor_dist_km": "Distance to Nearest Corridor (km)",
            "movement_score": "Movement Score",
        },
    )
    fig.update_layout(**PLOTLY_LAYOUT, xaxis=dict(**_AX), yaxis=dict(**_AX), height=420)
    return fig


def movement_seasonal_chart(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of movement score by season and species."""
    grp = df.groupby(["season", "species"])["movement_score"].mean().reset_index()
    fig = px.bar(
        grp, x="season", y="movement_score", color="species",
        barmode="group",
        title="Seasonal Movement Patterns by Species",
        labels={"movement_score": "Avg Movement Score", "season": "Season"},
    )
    fig.update_layout(**PLOTLY_LAYOUT, xaxis=dict(**_AX), yaxis=dict(**_AX), height=400,
                      legend=dict(bgcolor=PALETTE["surface"]))
    return fig


def movement_timeline_chart(df: pd.DataFrame) -> go.Figure:
    """Area chart showing movement score distribution across 24h."""
    hourly = df.groupby("hour").agg(
        avg_movement=("movement_score", "mean"),
        high_movement=("movement_score", lambda x: (x > 0.4).mean()),
        dawn_dusk_pct=("dawn_dusk", "mean"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["avg_movement"],
        mode="lines+markers", name="Avg Movement Score",
        line=dict(color=PALETTE["accent1"], width=3),
        marker=dict(size=8),
        fill="tozeroy", fillcolor="rgba(0,229,255,0.1)",
    ))
    fig.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly["high_movement"],
        mode="lines+markers", name="High Movement Rate (>0.4)",
        line=dict(color=PALETTE["high"], width=2, dash="dot"),
        marker=dict(size=6),
    ))
    # Dawn/dusk band
    fig.add_vrect(x0=5, x1=7, fillcolor="rgba(255,176,32,0.08)",
                  line_width=0, annotation_text="Dawn", annotation_position="top left")
    fig.add_vrect(x0=17, x1=19, fillcolor="rgba(255,176,32,0.08)",
                  line_width=0, annotation_text="Dusk", annotation_position="top left")

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="24-Hour Animal Movement Activity Timeline",
        xaxis=dict(title="Hour of Day", dtick=2, **_AX),
        yaxis=dict(title="Score / Rate", **_AX),
        legend=dict(x=0.01, y=0.99, bgcolor=PALETTE["surface"]),
        height=380,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# NDVI PREDICTION INDEX CHARTS
# ══════════════════════════════════════════════════════════════════════════════

NDVI_ZONES = [
    (0.0,  0.15, "Barren",      "#8B4513", "No vegetation"),
    (0.15, 0.30, "Sparse",      "#DAA520", "Grassland / degraded"),
    (0.30, 0.50, "Moderate",    "#9ACD32", "Mixed vegetation"),
    (0.50, 0.70, "Dense",       "#228B22", "Dense forest canopy"),
    (0.70, 1.00, "Very Dense",  "#006400", "Primary forest"),
]


def ndvi_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Histogram of NDVI values with zone annotations."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["ndvi"], nbinsx=50, name="NDVI Distribution",
        marker_color=PALETTE["accent2"], opacity=0.8,
    ))
    for lo, hi, label, color, _ in NDVI_ZONES:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=color, opacity=0.08,
                      line_width=0, annotation_text=label,
                      annotation_position="top left",
                      annotation_font_size=9)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="NDVI Distribution with Vegetation Zones",
        xaxis=dict(title="NDVI Value", **_AX),
        yaxis=dict(title="Count", **_AX),
        height=380,
    )
    return fig


def ndvi_seasonal_trend(df: pd.DataFrame) -> go.Figure:
    """Box plot of NDVI by season showing seasonal variation."""
    season_order = ["summer", "monsoon", "post_monsoon", "winter"]
    season_colors = {
        "summer": PALETTE["orange"],
        "monsoon": PALETTE["accent2"],
        "post_monsoon": PALETTE["accent3"],
        "winter": PALETTE["blue"],
    }
    fig = go.Figure()
    for s in season_order:
        s_data = df[df["season"] == s]["ndvi"]
        fig.add_trace(go.Box(
            y=s_data, name=s.replace("_", " ").title(),
            marker_color=season_colors.get(s, PALETTE["text"]),
            boxmean=True,
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="NDVI Seasonal Variation (Vegetation Health Forecast)",
        yaxis=dict(title="NDVI Value", **_AX),
        xaxis=dict(**_AX),
        height=400,
    )
    return fig


def ndvi_risk_correlation(df: pd.DataFrame) -> go.Figure:
    """Heatmap of NDVI bins vs risk score bins."""
    df_c = df.copy()
    df_c["ndvi_bin"] = pd.cut(df_c["ndvi"], bins=8, labels=[
        "0.00-0.12", "0.12-0.25", "0.25-0.38", "0.38-0.50",
        "0.50-0.62", "0.62-0.75", "0.75-0.88", "0.88-1.00",
    ])
    df_c["risk_bin"] = pd.cut(df_c["risk_score"], bins=5, labels=[
        "Very Low", "Low", "Medium", "High", "Critical",
    ])
    pivot = df_c.groupby(["ndvi_bin", "risk_bin"], observed=True).size().unstack(fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale=[[0, "#0a1628"], [0.5, "#FFD166"], [1, "#EF476F"]],
        showscale=True,
        hovertemplate="NDVI: %{y}<br>Risk: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="NDVI vs Risk Level Correlation Matrix",
        xaxis=dict(title="Risk Level", **_AX),
        yaxis=dict(title="NDVI Range", **_AX),
        height=420,
    )
    return fig


def ndvi_species_habitat(df: pd.DataFrame) -> go.Figure:
    """Violin plot of NDVI by species showing habitat preference."""
    fig = go.Figure()
    species_list = sorted(df["species"].unique())
    colors = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"],
              PALETTE["high"], PALETTE["critical"], PALETTE["blue"],
              PALETTE["purple"], PALETTE["orange"]]
    for i, sp in enumerate(species_list):
        sp_data = df[df["species"] == sp]
        fig.add_trace(go.Violin(
            y=sp_data["ndvi"], name=sp.capitalize(),
            box_visible=True, meanline_visible=True,
            fillcolor=colors[i % len(colors)],
            line_color=colors[i % len(colors)],
            opacity=0.7,
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Species Habitat Preference (NDVI Distribution)",
        yaxis=dict(title="NDVI Value", **_AX),
        xaxis=dict(**_AX),
        height=420,
    )
    return fig


def ndvi_prediction_gauge(ndvi_val: float) -> go.Figure:
    """Gauge showing current NDVI value with zone classification."""
    zone_label = "Unknown"
    zone_color = PALETTE["muted"]
    for lo, hi, label, color, _ in NDVI_ZONES:
        if lo <= ndvi_val < hi or (ndvi_val >= 0.70 and hi == 1.00):
            zone_label = label
            zone_color = color
            break

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=ndvi_val,
        number={"font": {"size": 48, "color": zone_color}},
        gauge={
            "axis": {"range": [0, 1], "tickfont": {"color": PALETTE["text"]}},
            "bar": {"color": zone_color},
            "bgcolor": PALETTE["surface"],
            "bordercolor": PALETTE["border"],
            "steps": [
                {"range": [0.0,  0.15], "color": "#1a0e05"},
                {"range": [0.15, 0.30], "color": "#1a1505"},
                {"range": [0.30, 0.50], "color": "#121a08"},
                {"range": [0.50, 0.70], "color": "#081a0a"},
                {"range": [0.70, 1.00], "color": "#051a05"},
            ],
        },
        title={"text": f"{zone_label} Vegetation",
               "font": {"color": zone_color, "size": 16}},
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=260)
    return fig


def ndvi_road_type_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of average NDVI by road type."""
    grp = df.groupby("road_type")["ndvi"].agg(["mean", "std"]).reset_index()
    grp = grp.sort_values("mean", ascending=True)
    colors = [PALETTE["accent2"] if v > 0.5 else PALETTE["accent3"] if v > 0.3
              else PALETTE["high"] for v in grp["mean"]]
    fig = go.Figure(go.Bar(
        x=grp["mean"], y=grp["road_type"], orientation="h",
        marker_color=colors,
        error_x=dict(type="data", array=grp["std"].tolist(), visible=True,
                     color=PALETTE["muted"]),
        text=[f"{v:.3f}" for v in grp["mean"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Avg NDVI: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Average NDVI by Road Type",
        xaxis=dict(title="Mean NDVI", **_AX),
        yaxis=dict(**_AX),
        height=360,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ALERT SYSTEM HELPERS
# ══════════════════════════════════════════════════════════════════════════════

ALERT_SEVERITY = {
    "CRITICAL": {"color": "#d500f9", "icon": "🚨", "priority": 4},
    "HIGH":     {"color": "#ff3d5a", "icon": "🔴", "priority": 3},
    "MEDIUM":   {"color": "#ffb020", "icon": "🟡", "priority": 2},
    "LOW":      {"color": "#00e676", "icon": "🟢", "priority": 1},
}


def generate_alerts(df: pd.DataFrame,
                    risk_threshold: float = 0.65,
                    corridor_threshold: float = 2.0,
                    visibility_threshold: int = 300,
                    speed_ratio_threshold: float = 1.2) -> list:
    """Generate alerts from dataset based on configurable thresholds."""
    alerts = []

    # High risk score alerts
    high_risk = df[df["risk_score"] > risk_threshold]
    if len(high_risk) > 0:
        for _, row in high_risk.head(50).iterrows():
            severity = "CRITICAL" if row["risk_score"] > 0.85 else "HIGH"
            alerts.append({
                "severity": severity,
                "type": "High Risk Zone",
                "message": f"Risk score {row['risk_score']:.1%} detected near ({row['latitude']:.3f}, {row['longitude']:.3f})",
                "species": row.get("species", "unknown"),
                "road_type": row.get("road_type", "unknown"),
                "risk_score": row["risk_score"],
                "lat": row["latitude"],
                "lon": row["longitude"],
                "hour": int(row.get("hour", 0)),
                "segment": row.get("highway_segment", "Unknown"),
            })

    # Night driving in high-risk areas
    night_risk = df[(df["night_flag"] == 1) & (df["risk_score"] > 0.5)]
    if len(night_risk) > 0:
        for _, row in night_risk.head(20).iterrows():
            alerts.append({
                "severity": "HIGH",
                "type": "Nocturnal Activity",
                "message": f"Night driving in wildlife zone at hour {int(row['hour']):02d}:00 — elevated animal crossing risk",
                "species": row.get("species", "unknown"),
                "road_type": row.get("road_type", "unknown"),
                "risk_score": row["risk_score"],
                "lat": row["latitude"],
                "lon": row["longitude"],
                "hour": int(row.get("hour", 0)),
                "segment": row.get("highway_segment", "Unknown"),
            })

    # Corridor proximity alerts
    near_corridor = df[df["corridor_dist_km"] < corridor_threshold]
    if len(near_corridor) > 0:
        for _, row in near_corridor.head(20).iterrows():
            severity = "HIGH" if row["corridor_dist_km"] < 1.0 else "MEDIUM"
            alerts.append({
                "severity": severity,
                "type": "Corridor Proximity",
                "message": f"Vehicle {row['corridor_dist_km']:.1f}km from wildlife corridor — migration path crossing likely",
                "species": row.get("species", "unknown"),
                "road_type": row.get("road_type", "unknown"),
                "risk_score": row["risk_score"],
                "lat": row["latitude"],
                "lon": row["longitude"],
                "hour": int(row.get("hour", 0)),
                "segment": row.get("highway_segment", "Unknown"),
            })

    # Low visibility alerts
    low_vis = df[df["visibility_m"] < visibility_threshold]
    if len(low_vis) > 0:
        for _, row in low_vis.head(15).iterrows():
            alerts.append({
                "severity": "MEDIUM",
                "type": "Low Visibility",
                "message": f"Visibility at {int(row['visibility_m'])}m — fog/rain impairing driver sightlines",
                "species": row.get("species", "unknown"),
                "road_type": row.get("road_type", "unknown"),
                "risk_score": row["risk_score"],
                "lat": row["latitude"],
                "lon": row["longitude"],
                "hour": int(row.get("hour", 0)),
                "segment": row.get("highway_segment", "Unknown"),
            })

    # Speeding alerts
    speeding = df[df["speed_ratio"] > speed_ratio_threshold]
    if len(speeding) > 0:
        for _, row in speeding.head(15).iterrows():
            severity = "HIGH" if row["speed_ratio"] > 1.4 else "MEDIUM"
            alerts.append({
                "severity": severity,
                "type": "Speed Violation",
                "message": f"Speed ratio {row['speed_ratio']:.2f}x — exceeding limit by {(row['speed_ratio']-1)*100:.0f}%",
                "species": row.get("species", "unknown"),
                "road_type": row.get("road_type", "unknown"),
                "risk_score": row["risk_score"],
                "lat": row["latitude"],
                "lon": row["longitude"],
                "hour": int(row.get("hour", 0)),
                "segment": row.get("highway_segment", "Unknown"),
            })

    # Breeding season alerts
    breeding = df[(df["breeding_season"] == 1) & (df["movement_score"] > 0.4)]
    if len(breeding) > 0:
        for _, row in breeding.head(15).iterrows():
            alerts.append({
                "severity": "MEDIUM",
                "type": "Breeding Season",
                "message": f"Breeding season with high movement ({row['movement_score']:.3f}) — increased animal activity",
                "species": row.get("species", "unknown"),
                "road_type": row.get("road_type", "unknown"),
                "risk_score": row["risk_score"],
                "lat": row["latitude"],
                "lon": row["longitude"],
                "hour": int(row.get("hour", 0)),
                "segment": row.get("highway_segment", "Unknown"),
            })

    # Sort by priority
    alerts.sort(key=lambda a: ALERT_SEVERITY[a["severity"]]["priority"], reverse=True)
    return alerts


def alert_severity_chart(alerts: list) -> go.Figure:
    """Donut chart of alert severity distribution."""
    counts = {}
    for a in alerts:
        sev = a["severity"]
        counts[sev] = counts.get(sev, 0) + 1

    labels = list(counts.keys())
    values = list(counts.values())
    colors = [ALERT_SEVERITY[s]["color"] for s in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=PALETTE["bg"], width=2)),
        textinfo="label+value",
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Alert Severity Distribution",
        showlegend=True,
        legend=dict(bgcolor=PALETTE["surface"]),
        height=360,
    )
    return fig


def alert_type_chart(alerts: list) -> go.Figure:
    """Horizontal bar chart of alert counts by type."""
    type_counts = {}
    for a in alerts:
        t = a["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    types = [t[0] for t in sorted_types][::-1]
    counts = [t[1] for t in sorted_types][::-1]

    fig = go.Figure(go.Bar(
        x=counts, y=types, orientation="h",
        marker_color=PALETTE["accent1"],
        text=counts, textposition="outside",
        hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Alerts by Category",
        xaxis=dict(title="Alert Count", **_AX),
        yaxis=dict(**_AX),
        height=380,
    )
    return fig


def alert_hourly_chart(alerts: list) -> go.Figure:
    """Bar chart showing alert distribution across 24 hours."""
    hour_counts = [0] * 24
    hour_severity = {h: {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0} for h in range(24)}
    for a in alerts:
        h = a.get("hour", 0)
        hour_counts[h] += 1
        hour_severity[h][a["severity"]] += 1

    fig = go.Figure()
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        vals = [hour_severity[h][sev] for h in range(24)]
        if sum(vals) > 0:
            fig.add_trace(go.Bar(
                x=list(range(24)), y=vals, name=sev,
                marker_color=ALERT_SEVERITY[sev]["color"],
            ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Alert Distribution by Hour of Day",
        barmode="stack",
        xaxis=dict(title="Hour", dtick=2, **_AX),
        yaxis=dict(title="Alert Count", **_AX),
        legend=dict(bgcolor=PALETTE["surface"]),
        height=380,
    )
    return fig


def alert_species_chart(alerts: list) -> go.Figure:
    """Bar chart of alerts by species."""
    sp_counts = {}
    for a in alerts:
        sp = a.get("species", "unknown").capitalize()
        sp_counts[sp] = sp_counts.get(sp, 0) + 1

    sorted_sp = sorted(sp_counts.items(), key=lambda x: x[1], reverse=True)
    species = [s[0] for s in sorted_sp]
    counts = [s[1] for s in sorted_sp]

    colors = [PALETTE["critical"] if c > np.median(counts) else PALETTE["high"]
              if c > np.median(counts) * 0.5 else PALETTE["accent3"]
              for c in counts]

    fig = go.Figure(go.Bar(
        x=species, y=counts,
        marker_color=colors,
        text=counts, textposition="outside",
        hovertemplate="<b>%{x}</b><br>Alerts: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Alerts by Species",
        xaxis=dict(**_AX),
        yaxis=dict(title="Alert Count", **_AX),
        height=360,
    )
    return fig
