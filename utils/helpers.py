"""
Utility functions — KDE hotspot density, GIS helpers, risk formatters
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional


# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "bg"        : "#0D1117",
    "surface"   : "#161B22",
    "border"    : "#30363D",
    "accent1"   : "#FF6B35",
    "accent2"   : "#00D4AA",
    "accent3"   : "#FFD166",
    "low"       : "#06D6A0",
    "medium"    : "#FFD166",
    "high"      : "#EF476F",
    "critical"  : "#B5179E",
    "text"      : "#E6EDF3",
    "muted"     : "#8B949E",
}

RISK_LEVELS = [
    (0.0,  0.25, "Low",      PALETTE["low"],      "🟢"),
    (0.25, 0.50, "Moderate", PALETTE["medium"],   "🟡"),
    (0.50, 0.75, "High",     PALETTE["high"],     "🔴"),
    (0.75, 1.00, "Critical", PALETTE["critical"], "🟣"),
]

# ── PLOTLY_LAYOUT: NO xaxis/yaxis — set them per chart to avoid conflicts ──────
PLOTLY_LAYOUT = dict(
    paper_bgcolor = PALETTE["bg"],
    plot_bgcolor  = PALETTE["surface"],
    font          = dict(color=PALETTE["text"], family="'JetBrains Mono', monospace"),
    margin        = dict(l=40, r=20, t=50, b=40),
)

# Reusable axis style — spread into xaxis/yaxis per chart
_AX = dict(gridcolor=PALETTE["border"], zerolinecolor=PALETTE["border"])


def get_risk_level(prob: float) -> dict:
    for lo, hi, label, color, emoji in RISK_LEVELS:
        if lo <= prob < hi or (prob >= 0.75 and hi == 1.00):
            return {"label": label, "color": color, "emoji": emoji, "prob": prob}
    return {"label": "Critical", "color": PALETTE["critical"], "emoji": "🟣", "prob": prob}


# ── KDE Hotspot Layer ─────────────────────────────────────────────────────────
def compute_kde_grid(lats: np.ndarray, lons: np.ndarray,
                     weights: Optional[np.ndarray] = None,
                     grid_size: int = 80) -> tuple:
    lat_grid  = np.linspace(lats.min() - 0.2, lats.max() + 0.2, grid_size)
    lon_grid  = np.linspace(lons.min() - 0.2, lons.max() + 0.2, grid_size)
    ll, lg    = np.meshgrid(lat_grid, lon_grid)
    positions = np.vstack([ll.ravel(), lg.ravel()])
    values    = np.vstack([lats, lons])
    kernel    = gaussian_kde(values, weights=weights, bw_method=0.15)
    density   = kernel(positions).reshape(grid_size, grid_size)
    return lat_grid, lon_grid, density


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
            "axis"       : {"range": [0, 100], "tickfont": {"color": PALETTE["text"]}},
            "bar"        : {"color": rl["color"]},
            "bgcolor"    : PALETTE["surface"],
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


def feature_importance_chart(shap_importance: list) -> go.Figure:
    top    = sorted(shap_importance, key=lambda x: x["shap_mean"], reverse=True)[:20]
    feat   = [x["feature"] for x in top][::-1]
    vals   = [x["shap_mean"] for x in top][::-1]
    colors = [PALETTE["accent1"] if v > np.median(vals) else PALETTE["accent2"] for v in vals]

    fig = go.Figure(go.Bar(
        x             = vals,
        y             = feat,
        orientation   = "h",
        marker        = dict(color=colors, line=dict(width=0)),
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


def hourly_risk_chart(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("hour")["accident"].agg(["mean", "count"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x            = grp["hour"],
        y            = grp["count"],
        name         = "Event Count",
        marker_color = PALETTE["accent2"],
        opacity      = 0.5,
        yaxis        = "y2",
    ))
    fig.add_trace(go.Scatter(
        x      = grp["hour"],
        y      = grp["mean"],
        mode   = "lines+markers",
        name   = "Accident Rate",
        line   = dict(color=PALETTE["accent1"], width=2.5),
        marker = dict(size=8),
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
        x             = grp["mean"],
        y             = grp["species"],
        orientation   = "h",
        marker_color  = colors,
        text          = [f"{v:.1%}" for v in grp["mean"]],
        textposition  = "outside",
        hovertemplate = "<b>%{y}</b><br>Rate: %{x:.1%}<br>Count: %{customdata}<extra></extra>",
        customdata    = grp["count"],
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
    pivot = df.pivot_table(
        index   = "season",
        columns = "road_type",
        values  = "accident",
        aggfunc = "mean"
    )
    fig = go.Figure(go.Heatmap(
        z            = pivot.values,
        x            = list(pivot.columns),
        y            = list(pivot.index),
        colorscale   = [[0, PALETTE["low"]], [0.5, PALETTE["medium"]], [1, PALETTE["critical"]]],
        text         = [[f"{v:.1%}" for v in row] for row in pivot.values],
        texttemplate = "%{text}",
        showscale    = True,
        hoverongaps  = False,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title  = "Accident Rate: Season × Road Type",
        xaxis  = dict(**_AX),
        yaxis  = dict(**_AX),
        height = 360,
    )
    return fig


def shap_waterfall(shap_vals: np.ndarray, feature_names: list,
                   feature_values: np.ndarray, base_value: float) -> go.Figure:
    pairs = sorted(zip(shap_vals, feature_names, feature_values),
                   key=lambda x: abs(x[0]), reverse=True)[:15]
    sv, fn, fv = zip(*pairs)
    sv     = list(sv)[::-1]
    fn     = [f"{n}={v:.2f}" if isinstance(v, float) else f"{n}={v}"
              for n, v in zip(fn, fv)][::-1]
    colors = [PALETTE["high"] if s > 0 else PALETTE["accent2"] for s in sv]

    fig = go.Figure(go.Bar(
        x             = sv,
        y             = fn,
        orientation   = "h",
        marker_color  = colors,
        hovertemplate = "<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
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
    df2  = df.copy()
    roll = df2["accident"].rolling(200, min_periods=50).mean().reset_index(drop=True)
    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        y         = roll,
        mode      = "lines",
        line      = dict(color=PALETTE["accent1"], width=2),
        name      = "Rolling accident rate",
        fill      = "tozeroy",
        fillcolor = "rgba(255,107,53,0.15)",
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
    fig    = px.scatter(
        sample,
        x         = "ndvi",
        y         = "risk_score",
        color     = "species",
        size      = "past_accidents",
        size_max  = 14,
        opacity   = 0.65,
        title     = "NDVI vs Risk Score by Species",
        labels    = {"ndvi": "Forest Density (NDVI)", "risk_score": "Risk Score"},
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis  = dict(**_AX),
        yaxis  = dict(**_AX),
        height = 420,
    )
    return fig


def confusion_matrix_chart(cm: list) -> go.Figure:
    labels = ["No Accident", "Accident"]
    fig    = go.Figure(go.Heatmap(
        z            = cm,
        x            = labels,
        y            = labels,
        colorscale   = [[0, PALETTE["surface"]], [1, PALETTE["accent1"]]],
        text         = cm,
        texttemplate = "%{text}",
        showscale    = False,
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title  = "Confusion Matrix",
        xaxis  = dict(title="Predicted", **_AX),
        yaxis  = dict(title="Actual", **_AX),
        height = 300,
    )
    return fig