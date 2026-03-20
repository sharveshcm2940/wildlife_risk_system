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

/* ═══════════════════════════════════════════════════════════════════════════
   1. RADAR SWEEP — sidebar animated ping
   ═══════════════════════════════════════════════════════════════════════════ */
@keyframes radar-sweep {
  0%   { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
.radar-container {
  position: relative;
  width: 70px; height: 70px;
  margin: 0 auto 0.5rem;
}
.radar-ring {
  position: absolute; inset: 0;
  border: 1px solid rgba(0,229,255,0.15);
  border-radius: 50%;
}
.radar-ring.inner { inset: 12px; border-color: rgba(0,229,255,0.1); }
.radar-ring.core  { inset: 24px; border-color: rgba(0,229,255,0.08); }
.radar-sweep-line {
  position: absolute;
  top: 50%; left: 50%;
  width: 50%; height: 1px;
  background: linear-gradient(90deg, var(--cyan), transparent);
  transform-origin: 0 0;
  animation: radar-sweep 4s linear infinite;
  opacity: 0.7;
}
.radar-sweep-line::after {
  content: '';
  position: absolute;
  right: 0; top: -3px;
  width: 6px; height: 6px;
  background: var(--cyan);
  border-radius: 50%;
  box-shadow: 0 0 8px var(--cyan);
}
.radar-center {
  position: absolute;
  top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  width: 4px; height: 4px;
  background: var(--cyan);
  border-radius: 50%;
  box-shadow: 0 0 12px var(--cyan);
}

/* ═══════════════════════════════════════════════════════════════════════════
   2. CLASSIFIED TICKER BAR — scrolling top banner
   ═══════════════════════════════════════════════════════════════════════════ */
@keyframes ticker-scroll {
  0%   { transform: translateX(100%); }
  100% { transform: translateX(-100%); }
}
.classified-ticker {
  overflow: hidden;
  background: linear-gradient(90deg, transparent, rgba(0,229,255,0.04), transparent);
  border: 1px solid var(--border);
  border-left: none; border-right: none;
  height: 18px;
  margin-bottom: 0.8rem;
  position: relative;
}
.classified-ticker::before,
.classified-ticker::after {
  content: '';
  position: absolute;
  top: 0; bottom: 0;
  width: 40px;
  z-index: 1;
}
.classified-ticker::before { left: 0; background: linear-gradient(90deg, var(--bg), transparent); }
.classified-ticker::after  { right: 0; background: linear-gradient(90deg, transparent, var(--bg)); }
.ticker-text {
  display: inline-block;
  white-space: nowrap;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.48rem;
  color: var(--muted-lo);
  letter-spacing: 0.35em;
  text-transform: uppercase;
  animation: ticker-scroll 30s linear infinite;
  line-height: 18px;
}

/* ═══════════════════════════════════════════════════════════════════════════
   3. CROSSHAIR TARGETS on hover — tactical focus
   ═══════════════════════════════════════════════════════════════════════════ */
.metric-card:hover::before {
  width: 16px; height: 16px;
  border-color: var(--cyan);
  filter: drop-shadow(0 0 3px var(--cyan));
  transition: all 0.2s ease;
}
.metric-card:hover::after {
  width: 16px; height: 16px;
  border-color: var(--cyan);
  filter: drop-shadow(0 0 3px var(--cyan));
  transition: all 0.2s ease;
}
.metric-card:hover {
  border-color: var(--border-hi);
  box-shadow: 0 0 15px rgba(0,229,255,0.06);
  transition: all 0.2s ease;
}

/* ═══════════════════════════════════════════════════════════════════════════
   4. THREAT LEVEL STRIP — colored edge indicator
   ═══════════════════════════════════════════════════════════════════════════ */
.threat-strip {
  display: flex; gap: 1px;
  margin: 0.6rem 0;
  height: 3px;
}
.threat-strip .seg {
  flex: 1;
  transition: all 0.3s ease;
}
.threat-strip .seg:hover { height: 6px; margin-top: -1.5px; }

/* ═══════════════════════════════════════════════════════════════════════════
   5. MATRIX DATA RAIN — background decoration
   ═══════════════════════════════════════════════════════════════════════════ */
@keyframes rain-fall {
  0%   { transform: translateY(-100%); opacity: 0; }
  10%  { opacity: 1; }
  90%  { opacity: 1; }
  100% { transform: translateY(400%); opacity: 0; }
}
.data-rain {
  position: fixed;
  top: 0; right: 20px;
  width: 200px; height: 100vh;
  pointer-events: none;
  z-index: 0;
  overflow: hidden;
}
.rain-col {
  position: absolute;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.45rem;
  color: rgba(0,229,255,0.06);
  writing-mode: vertical-rl;
  animation: rain-fall linear infinite;
  letter-spacing: 0.3em;
}

/* ═══════════════════════════════════════════════════════════════════════════
   6. HOLOGRAPHIC SHIMMER — section shine effect
   ═══════════════════════════════════════════════════════════════════════════ */
@keyframes holo-shimmer {
  0%   { left: -150%; }
  100% { left: 250%; }
}
.section-header::after {
  content: '';
  position: absolute;
  top: 0; bottom: 0;
  width: 60px;
  background: linear-gradient(90deg, transparent, rgba(0,229,255,0.06), transparent);
  animation: holo-shimmer 6s ease-in-out infinite;
}

/* ═══════════════════════════════════════════════════════════════════════════
   7. CORNER VIGNETTE — ambient darkening
   ═══════════════════════════════════════════════════════════════════════════ */
.stApp::after {
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: radial-gradient(ellipse at center, transparent 60%, rgba(0,0,0,0.4) 100%);
  pointer-events: none;
  z-index: 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
   8. TACTICAL GRID — dot matrix background
   ═══════════════════════════════════════════════════════════════════════════ */
.block-container::before {
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background-image:
    radial-gradient(circle, rgba(0,229,255,0.02) 1px, transparent 1px);
  background-size: 28px 28px;
  pointer-events: none;
  z-index: 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
   9. BREATHING RING — branding pulse around logo
   ═══════════════════════════════════════════════════════════════════════════ */
@keyframes breathe-ring {
  0%, 100% { box-shadow: 0 0 0 0 rgba(0,229,255,0.3); }
  50%      { box-shadow: 0 0 0 8px rgba(0,229,255,0); }
}
.brand-ring {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 52px; height: 52px;
  border: 1px solid rgba(0,229,255,0.25);
  border-radius: 50%;
  animation: breathe-ring 3s ease-in-out infinite;
}

/* ═══════════════════════════════════════════════════════════════════════════
   10. GLITCH FLICKER — header text distortion
   ═══════════════════════════════════════════════════════════════════════════ */
@keyframes glitch {
  0%, 98% { text-shadow: 0 0 20px rgba(0,229,255,0.2); }
  99%     { text-shadow: -2px 0 #ff3d5a, 2px 0 #00e5ff; }
  100%    { text-shadow: 0 0 20px rgba(0,229,255,0.2); }
}
.stMarkdown h1 span[style*="color:#00e5ff"],
.stMarkdown h1 span[style*="color:#00e676"] {
  animation: glitch 8s ease-in-out infinite;
}

/* ═══════════════════════════════════════════════════════════════════════════
   11. BOOT SEQUENCE — typing cursor on classified labels
   ═══════════════════════════════════════════════════════════════════════════ */
@keyframes blink-cursor {
  0%, 100% { border-right-color: transparent; }
  50%      { border-right-color: var(--cyan); }
}
.boot-text {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.5rem;
  color: var(--muted-lo);
  letter-spacing: 0.2em;
  border-right: 1.5px solid var(--cyan);
  padding-right: 3px;
  animation: blink-cursor 1.2s step-end infinite;
  display: inline-block;
}

/* ═══════════════════════════════════════════════════════════════════════════
   12. TACTICAL SCOPE RETICLE — sidebar brand element
   ═══════════════════════════════════════════════════════════════════════════ */
.reticle {
  position: relative;
  width: 80px; height: 80px;
  margin: 0 auto;
}
.reticle::before, .reticle::after {
  content: '';
  position: absolute;
  background: rgba(0,229,255,0.15);
}
.reticle::before {
  top: 50%; left: 8px; right: 8px;
  height: 1px;
  transform: translateY(-50%);
}
.reticle::after {
  left: 50%; top: 8px; bottom: 8px;
  width: 1px;
  transform: translateX(-50%);
}
.reticle-outer {
  position: absolute; inset: 0;
  border: 1px solid rgba(0,229,255,0.12);
  border-radius: 50%;
}
.reticle-inner {
  position: absolute; inset: 14px;
  border: 1px dashed rgba(0,229,255,0.08);
  border-radius: 50%;
}
.reticle-dot {
  position: absolute;
  top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  width: 5px; height: 5px;
  background: var(--cyan);
  border-radius: 50%;
  box-shadow: 0 0 10px var(--cyan), 0 0 20px rgba(0,229,255,0.3);
}
.reticle-tick {
  position: absolute;
  background: rgba(0,229,255,0.2);
}
.reticle-tick.t { top: 2px; left: 50%; width: 1px; height: 6px; transform: translateX(-50%); }
.reticle-tick.b { bottom: 2px; left: 50%; width: 1px; height: 6px; transform: translateX(-50%); }
.reticle-tick.l { left: 2px; top: 50%; height: 1px; width: 6px; transform: translateY(-50%); }
.reticle-tick.r { right: 2px; top: 50%; height: 1px; width: 6px; transform: translateY(-50%); }

/* ═══════════════════════════════════════════════════════════════════════════
   BONUS: Intel panel styling
   ═══════════════════════════════════════════════════════════════════════════ */
.intel-panel {
  background: linear-gradient(145deg, var(--surface), var(--bg-alt));
  border: 1px solid var(--border);
  padding: 0.8rem 1rem;
  position: relative;
  margin: 0.5rem 0;
  overflow: hidden;
}
.intel-panel::before {
  content: '';
  position: absolute;
  top: 0; left: 0; width: 100%; height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan-dim), transparent);
}
.intel-panel .panel-label {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.5rem;
  color: var(--muted-lo);
  letter-spacing: 0.2em;
  text-transform: uppercase;
  margin-bottom: 0.3rem;
}
.intel-panel .panel-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  color: var(--cyan);
}

</style>
""", unsafe_allow_html=True)

# ═══════════════════ SPLASH SCREEN ═══════════════════
if "splash_done" not in st.session_state:
    st.session_state.splash_done = True
    st.markdown("""<div id="spl" style="position:fixed;inset:0;z-index:999999;background:#030508;display:flex;flex-direction:column;align-items:center;justify-content:center;font-family:'JetBrains Mono',monospace;animation:sO .8s ease 4.5s forwards">
<style>
@keyframes sO{to{opacity:0;visibility:hidden;pointer-events:none}}
@keyframes sR{to{transform:rotate(360deg)}}
@keyframes sP{0%,100%{box-shadow:0 0 0 0 rgba(0,229,255,.4)}50%{box-shadow:0 0 0 14px rgba(0,229,255,0)}}
@keyframes sB{from{width:0}to{width:100%}}
@keyframes sL{from{opacity:0;transform:translateX(-6px)}to{opacity:1;transform:translateX(0)}}
.sl{font-size:.55rem;color:#3a4a5c;letter-spacing:.1em;margin:.13rem 0;opacity:0;animation:sL .25s ease forwards}
.sl .g{color:#00e676}.sl .w{color:#ffb020}.sl .c{color:#00e5ff}
.sb{width:250px;height:2px;background:#1a2332;margin:.2rem 0;overflow:hidden}
.sb div{height:100%;background:linear-gradient(90deg,#00e5ff,#4da6ff);animation:sB ease-out forwards}
</style>
<div style="position:relative;width:90px;height:90px;margin-bottom:1rem">
<div style="position:absolute;inset:0;border:1px solid rgba(0,229,255,.18);border-radius:50%"></div>
<div style="position:absolute;inset:15px;border:1px solid rgba(0,229,255,.1);border-radius:50%"></div>
<div style="position:absolute;inset:30px;border:1px dashed rgba(0,229,255,.06);border-radius:50%"></div>
<div style="position:absolute;top:50%;left:8px;right:8px;height:1px;background:rgba(0,229,255,.12);transform:translateY(-50%)"></div>
<div style="position:absolute;left:50%;top:8px;bottom:8px;width:1px;background:rgba(0,229,255,.12);transform:translateX(-50%)"></div>
<div style="position:absolute;top:50%;left:50%;width:50%;height:1px;background:linear-gradient(90deg,#00e5ff,transparent);transform-origin:0 0;animation:sR 3s linear infinite;opacity:.7"><div style="position:absolute;right:0;top:-3px;width:6px;height:6px;background:#00e5ff;border-radius:50%;box-shadow:0 0 10px #00e5ff"></div></div>
<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:5px;height:5px;background:#00e5ff;border-radius:50%;box-shadow:0 0 12px #00e5ff;animation:sP 2s ease-in-out infinite"></div>
</div>
<div style="font-size:1.6rem;font-weight:700;color:#00e5ff;letter-spacing:.18em;text-shadow:0 0 25px rgba(0,229,255,.35);margin-bottom:.1rem">WILDGUARD</div>
<div style="font-size:.42rem;color:#3a4a5c;letter-spacing:.35em;margin-bottom:1rem">THREAT INTELLIGENCE PLATFORM · v3.0</div>
<div style="width:280px;text-align:left">
<div class="sl" style="animation-delay:.2s">[<span class=g>OK</span>] KERNEL INIT · SECURE BOOT</div>
<div class="sl" style="animation-delay:.5s">[<span class=g>OK</span>] XGBOOST CLASSIFIER · <span class=c>500 ESTIMATORS</span></div>
<div class="sl" style="animation-delay:.8s">[<span class=g>OK</span>] RANDOM FOREST · <span class=c>400 TREES</span></div>
<div class="sb"><div style="animation-delay:.9s;animation-duration:1s"></div></div>
<div class="sl" style="animation-delay:1.1s">[<span class=g>OK</span>] SHAP TREE EXPLAINER · <span class=c>SHAPLEY ADDITIVE</span></div>
<div class="sl" style="animation-delay:1.5s">[<span class=g>OK</span>] REAL-TIME PIPELINE · <span class=c>7 DATA SOURCES</span></div>
<div class="sl" style="animation-delay:1.8s">[<span class=g>OK</span>] GEOSPATIAL ENGINE · FOLIUM HEATMAP</div>
<div class="sl" style="animation-delay:2.1s">[<span class=w>!!</span>] THREAT LEVEL · <span class=w>ELEVATED</span></div>
<div class="sb"><div style="animation-delay:2.2s;animation-duration:1.2s"></div></div>
<div class="sl" style="animation-delay:2.5s;font-size:.42rem">&nbsp;&nbsp;WEATHER · OSM · GBIF · NEWS · NDVI · TRAFFIC · GOV</div>
<div class="sl" style="animation-delay:2.9s">[<span class=g>OK</span>] 28 HIGHWAY SEGMENTS · SOUTH INDIA</div>
<div class="sl" style="animation-delay:3.3s">[<span class=g>::</span>] <span class=c>SYSTEM READY</span> · CLEARANCE LEVEL 4</div>
</div>
<div style="position:absolute;bottom:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#00e5ff,transparent);opacity:.3"></div>
<div style="position:absolute;bottom:6px;font-size:.35rem;color:#1a2332;letter-spacing:.25em">WILDLIFE-VEHICLE COLLISION RISK INTELLIGENCE · CLASSIFIED</div>
</div>
<script>setTimeout(function(){var e=document.getElementById('spl');if(e)e.style.display='none'},5200)</script>
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

# ── Sidebar — Command Console ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1.2rem 0 0.6rem 0;'>
      <div class='boot-text'>SYSTEM ONLINE · CLEARANCE LEVEL 4</div>
      <div class='reticle' style='margin: 0.6rem auto 0.2rem;'>
        <div class='reticle-outer'></div>
        <div class='reticle-inner'></div>
        <div class='reticle-dot'></div>
        <div class='reticle-tick t'></div>
        <div class='reticle-tick b'></div>
        <div class='reticle-tick l'></div>
        <div class='reticle-tick r'></div>
        <div class='radar-sweep-line'></div>
      </div>
      <div style='font-family:"JetBrains Mono",monospace; font-size:1.1rem; color:#00e5ff; font-weight:700; letter-spacing:0.12em; text-shadow: 0 0 18px rgba(0,229,255,0.35);'>WILDGUARD</div>
      <div style='font-family:"IBM Plex Mono",monospace; font-size:0.48rem; color:#3a4a5c; letter-spacing:0.25em; margin-top:0.1rem;'>THREAT INTELLIGENCE PLATFORM v3.0</div>
    </div>

    <!-- Threat Level Strip -->
    <div class='threat-strip' title='Threat Level: Elevated'>
      <div class='seg' style='background:#00e676;'></div>
      <div class='seg' style='background:#00e676;'></div>
      <div class='seg' style='background:#4caf50;'></div>
      <div class='seg' style='background:#ffb020;'></div>
      <div class='seg' style='background:#ffb020;'></div>
      <div class='seg' style='background:#ff7043;'></div>
      <div class='seg' style='background:#ff3d5a;'></div>
      <div class='seg' style='background:#ff3d5a;'></div>
      <div class='seg' style='background:#d500f9;'></div>
      <div class='seg' style='background:#1a2332;'></div>
    </div>

    <div style='display:flex; justify-content:space-between; padding:0 0.2rem;'>
      <div style='display:flex; align-items:center; gap:0.3rem;'>
        <span class='live-dot'></span>
        <span style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; color:#00e676; letter-spacing:0.12em;'>PIPELINE ACTIVE</span>
      </div>
      <span style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; color:#3a4a5c; letter-spacing:0.08em;'>▸ SECURE</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🎯 Risk Intel & Map",
         "📊 Analytics", "⚡ Future Hotspots", "🧠 Model Insights", "📡 Data Sources"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    xgb_auc = xgb_metrics.get('roc_auc', 0)
    rf_auc  = rf_metrics.get('roc_auc', 0)
    st.markdown(f"""
    <div style='padding:0.3rem 0;'>
      <div style='font-family:"IBM Plex Mono",monospace; font-size:0.48rem; color:#3a4a5c; letter-spacing:0.2em; margin-bottom:0.4rem;'>▸ DUAL-MODEL STATUS</div>

      <div class='intel-panel'>
        <div class='panel-label'>XGBoost Gradient Boost</div>
        <div style='display:flex; justify-content:space-between; align-items:baseline;'>
          <div class='panel-value'>AUC {xgb_auc:.4f}</div>
          <span style='font-size:0.5rem; color:#00e676;'>● TRAINED</span>
        </div>
      </div>

      <div class='intel-panel'>
        <div class='panel-label'>Random Forest Ensemble</div>
        <div style='display:flex; justify-content:space-between; align-items:baseline;'>
          <div class='panel-value' style='color:#4da6ff;'>AUC {rf_auc:.4f}</div>
          <span style='font-size:0.5rem; color:#00e676;'>● TRAINED</span>
        </div>
      </div>

      <div class='intel-panel'>
        <div class='panel-label'>Active Classifier</div>
        <div class='panel-value' style='color:#00e676;'>◆ {best_model}</div>
      </div>
    </div>

    <!-- System Intel -->
    <div style='margin-top:0.8rem;'>
      <div style='font-family:"IBM Plex Mono",monospace; font-size:0.48rem; color:#3a4a5c; letter-spacing:0.2em; margin-bottom:0.4rem;'>▸ SYSTEM INTEL</div>

      <div class='intel-panel'>
        <div class='panel-label'>Data Pipeline</div>
        <div style='display:flex; gap:0.4rem; flex-wrap:wrap; margin-top:0.2rem;'>
          <span style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; padding:0.1rem 0.4rem; border:1px solid rgba(0,229,255,0.2); color:var(--cyan);'>WEATHER</span>
          <span style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; padding:0.1rem 0.4rem; border:1px solid rgba(0,229,255,0.2); color:var(--cyan);'>OSM</span>
          <span style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; padding:0.1rem 0.4rem; border:1px solid rgba(0,229,255,0.2); color:var(--cyan);'>GBIF</span>
          <span style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; padding:0.1rem 0.4rem; border:1px solid rgba(0,229,255,0.2); color:var(--cyan);'>NEWS</span>
          <span style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; padding:0.1rem 0.4rem; border:1px solid rgba(0,229,255,0.2); color:var(--cyan);'>NDVI</span>
          <span style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; padding:0.1rem 0.4rem; border:1px solid rgba(0,229,255,0.2); color:var(--cyan);'>TRAFFIC</span>
          <span style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; padding:0.1rem 0.4rem; border:1px solid rgba(0,229,255,0.2); color:var(--cyan);'>GOV</span>
        </div>
      </div>

      <div class='intel-panel'>
        <div class='panel-label'>Feature Vector</div>
        <div class='panel-value'>{len(model.feature_cols)} dimensions</div>
      </div>

      <div class='intel-panel'>
        <div class='panel-label'>Training Records</div>
        <div class='panel-value'>{len(df):,} observations</div>
      </div>

      <div class='intel-panel'>
        <div class='panel-label'>SHAP Explainer</div>
        <div style='display:flex; justify-content:space-between; align-items:baseline;'>
          <div class='panel-value'>TreeSHAP</div>
          <span style='font-size:0.5rem; color:#00e676;'>● READY</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Global HUD Elements ──────────────────────────────────────────────────────
# Data Rain (Matrix falling characters — right edge)
rain_cols = ""
import random
data_chars = ["01001", "NDVI", "0.847", "RISK", "SHAP", "25.3C", "TIGER", "NH48", "ALERT", "KDE", "RF:0.81", "XGB"]
for i in range(8):
    left = 10 + i * 25
    dur = random.uniform(8, 18)
    delay = random.uniform(0, 8)
    text = " · ".join(random.choices(data_chars, k=6))
    rain_cols += f"<div class='rain-col' style='left:{left}px; animation-duration:{dur}s; animation-delay:{delay}s;'>{text}</div>"

st.markdown(f"""
<div class='data-rain'>{rain_cols}</div>

<!-- Classified Ticker -->
<div class='classified-ticker'>
  <span class='ticker-text'>
    ◈ WILDGUARD THREAT INTELLIGENCE — CLASSIFIED // RESTRICTED ACCESS — DUAL MODEL ENSEMBLE (XGBOOST + RANDOM FOREST) — 7 REAL-TIME DATA SOURCES — SHAP EXPLAINABILITY ENGINE ACTIVE — COVERING 28 HIGHWAY SEGMENTS ACROSS SOUTH INDIA — WILDLIFE CORRIDOR MONITORING — AUTOMATED RISK SCORING —
    ◈ WILDGUARD THREAT INTELLIGENCE — CLASSIFIED // RESTRICTED ACCESS — DUAL MODEL ENSEMBLE (XGBOOST + RANDOM FOREST) — 7 REAL-TIME DATA SOURCES — SHAP EXPLAINABILITY ENGINE ACTIVE —
  </span>
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
# PAGE 2 — RISK INTEL & MAP (Combined Predictor + Map)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Risk Intel & Map":
    from streamlit_folium import st_folium
    from utils.helpers import (
        dual_risk_gauge, shap_waterfall, get_risk_level,
        data_source_status_chart, build_folium_map, PALETTE
    )
    from data.realtime_extractor import RealtimeDataExtractor, geocode_place, PROTECTED_AREAS, WILDLIFE_CORRIDORS, _haversine
    import folium
    from folium.plugins import HeatMap, MarkerCluster

    # ── Tile Layers (using subdomains for speed) ──────────────────────────────
    GOOGLE_SAT   = "https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    GOOGLE_HYB   = "https://mt{s}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}"
    GOOGLE_ROAD  = "https://mt{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}"
    ESRI_SAT     = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    GOOGLE_SUBS  = ["0","1","2","3"]  # Parallel tile loading across 4 servers

    st.markdown("""
    <div style='margin-bottom:1rem;'>
      <div style='font-family:"IBM Plex Mono",monospace; font-size:0.5rem; color:#3a4a5c; letter-spacing:0.25em; margin-bottom:0.3rem;'>
        ▸ TACTICAL RISK ASSESSMENT // SEARCH → EXTRACT → PREDICT → MAP
      </div>
      <h1 style='font-family:"Inter",sans-serif; font-size:1.8rem; font-weight:600; margin:0 0 0.15rem 0; color:#e8f0fa;'>
        Risk <span style='color:#00e5ff;'>Intelligence</span> & Geospatial Map
      </h1>
      <p style='color:#5a6d82; font-size:0.78rem; margin:0; font-family:"IBM Plex Mono",monospace;'>
        Google Maps satellite · 7-source real-time pipeline · Dual-model ensemble · SHAP explainability
      </p>
    </div>
    """, unsafe_allow_html=True)

    # ── SECTION A: Location Search ────────────────────────────────────────────
    st.markdown("<div class='section-header'>📍 Search Location</div>", unsafe_allow_html=True)

    search_col, search_btn_col = st.columns([4, 1])
    with search_col:
        place_query = st.text_input(
            "🔍 Search any place in India",
            placeholder="e.g., Bandipur, Wayanad, Mudumalai, Mysore-Ooty Highway…",
            key="place_search"
        )
    with search_btn_col:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("🔎 Search", use_container_width=True)

    resolved_lat, resolved_lon = None, None
    resolved_name = ""
    sub_locations = []

    if search_clicked and place_query:
        with st.spinner(f"🌍 Geocoding '{place_query}' via Nominatim…"):
            geo_result = geocode_place(place_query)
        if geo_result and geo_result.get("found"):
            resolved_lat = geo_result["lat"]
            resolved_lon = geo_result["lon"]
            resolved_name = geo_result.get("display_name", "")
            st.success(f"📍 **Found:** {resolved_name}")
            st.markdown(f"<div style='font-size:0.78rem; color:var(--muted);'>Lat: {resolved_lat:.5f} | Lon: {resolved_lon:.5f} | Type: {geo_result.get('place_type','')}</div>", unsafe_allow_html=True)
            all_results = geo_result.get("all_results", [])
            if len(all_results) > 1:
                sub_locations = all_results
                st.markdown(f"<div style='font-size:0.72rem; color:#00e5ff; margin-top:0.3rem;'>🎯 Found **{len(all_results)} specific locations** — all highlighted on the map</div>", unsafe_allow_html=True)
                with st.expander(f"📌 View all {len(all_results)} matching locations"):
                    for i, r in enumerate(all_results):
                        st.markdown(f"**{i+1}.** {r['name']} — `{r['lat']:.4f}, {r['lon']:.4f}` · {r.get('type', '')}")
        else:
            st.error(f"❌ Could not find '{place_query}'. Try a more specific name.")

    st.markdown("<div style='text-align:center; color:var(--muted); font-size:0.75rem; margin:0.3rem 0;'>─── OR select a preset wildlife zone ───</div>", unsafe_allow_html=True)

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
        if resolved_lat is not None: default_lat = resolved_lat
        lat = st.number_input("Latitude", min_value=6.0, max_value=37.0, value=default_lat, step=0.01)
    with c3:
        if resolved_lon is not None: default_lon = resolved_lon
        lon = st.number_input("Longitude", min_value=68.0, max_value=97.0, value=default_lon, step=0.01)

    c4, c5, c6 = st.columns(3)
    with c4: speed_limit  = st.selectbox("Speed Limit (km/h)", [30, 40, 60, 80, 100], index=2)
    with c5: actual_speed = st.slider("Actual Speed (km/h)", 10, 140, 65)
    with c6: past_acc     = st.slider("Past Accident Count", 0, 30, 3)

    # ── Map Layer Controls ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🗺 Map Configuration</div>", unsafe_allow_html=True)
    ml1, ml2, ml3 = st.columns(3)
    with ml1:
        base_layer = st.selectbox("Base Map", ["OpenStreetMap (Fast)", "Google Satellite Hybrid", "Google Satellite", "Google Roads", "ESRI Satellite", "Dark (CartoDB)"], index=0)
    with ml2:
        show_heatmap = st.checkbox("Show Risk Heatmap", value=True)
        show_pas     = st.checkbox("Show Protected Areas", value=True)
    with ml3:
        show_corridors = st.checkbox("Show Wildlife Corridors", value=True)
        show_all_data  = st.checkbox("Show All Training Data", value=False)

    run_btn = st.button("⚡ EXTRACT DATA & PREDICT RISK", use_container_width=True)

    # ── SECTION B: EXECUTION ──────────────────────────────────────────────────
    if run_btn:
        with st.spinner("📡 Extracting real-time data from APIs, news sites & govt portals…"):
            extractor = RealtimeDataExtractor()
            feature_df = extractor.extract_all(lat, lon, speed_limit=speed_limit, actual_speed=actual_speed, past_accidents=past_acc)
            extraction_log = extractor.get_extraction_log()
            summary        = extractor.get_extraction_summary()

        spatial_data = None
        for er in extractor.extraction_log:
            if er.source_type == "computation" and "nearest_pa" in er.data:
                spatial_data = er.data
        st.session_state["last_prediction"] = {
            "lat": lat, "lon": lon, "location_name": resolved_name or preset,
            "features": feature_df.iloc[0].to_dict(), "log": extraction_log,
            "summary": summary, "spatial": spatial_data, "sub_locations": sub_locations,
        }

        # ── PREDICTION RESULTS ────────────────────────────────────────────────
        st.markdown("<div class='section-header'>🤖 Dual-Model Prediction Results</div>", unsafe_allow_html=True)
        if model.rf_model is not None:
            results = model.predict_both(feature_df)
            xgb_prob, rf_prob, avg_prob, agree = results["xgb_probability"], results["rf_probability"], results["avg_probability"], results["agreement"]
        else:
            xgb_prob = float(model.predict_risk(feature_df)[0])
            rf_prob = avg_prob = xgb_prob; agree = True

        rl_xgb, rl_rf, rl_avg = get_risk_level(xgb_prob), get_risk_level(rf_prob), get_risk_level(avg_prob)
        st.plotly_chart(dual_risk_gauge(xgb_prob, rf_prob), use_container_width=True)

        r1, r2, r3 = st.columns(3)
        for col, label, prob, rl in [(r1,"XGBoost",xgb_prob,rl_xgb),(r2,"Random Forest",rf_prob,rl_rf),(r3,f"Ensemble · {'✅ Agree' if agree else '⚠️ Diverge'}",avg_prob,rl_avg)]:
            with col:
                st.markdown(f"<div style='text-align:center;'><span class='risk-badge' style='background:{rl['color']}33; color:{rl['color']}; border:1.5px solid {rl['color']};'>{rl['emoji']} {label}: {prob:.1%}</span></div>", unsafe_allow_html=True)

        # ── DATA SOURCES ──────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>📡 Real-Time Extraction Status</div>", unsafe_allow_html=True)
        api_sources  = [l for l in extraction_log if l.get('source_type') == 'api']
        news_sources = [l for l in extraction_log if l.get('source_type') == 'news']
        govt_sources = [l for l in extraction_log if l.get('source_type') == 'government']

        st.markdown("**🌐 API Sources**")
        api_cols = st.columns(max(len(api_sources), 1))
        for i, le in enumerate(api_sources):
            with api_cols[i]:
                si = "✅" if le["status"]=="success" else "⚠️"
                st.markdown(f"<div class='source-card'><div class='source-name'>{si} {le['source']}</div><div style='font-size:0.72rem;'><span class='status-{le['status']}'>{le['status'].upper()}</span> · {le['response_ms']}ms</div></div>", unsafe_allow_html=True)

        ng1, ng2 = st.columns(2)
        for col, lbl, srcs in [(ng1,"📰 News",news_sources),(ng2,"🏛️ Govt",govt_sources)]:
            with col:
                st.markdown(f"**{lbl}**")
                for le in srcs:
                    si = "✅" if le["status"]=="success" else "⚠️"
                    st.markdown(f"<div class='source-card'><div class='source-name'>{si} {le['source']}</div><div style='font-size:0.72rem;'>{le['status'].upper()} · {le['response_ms']}ms</div></div>", unsafe_allow_html=True)

        st.plotly_chart(data_source_status_chart(extraction_log), use_container_width=True)

        # ── SHAP ──────────────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>🔬 SHAP Explanation — Why This Prediction</div>", unsafe_allow_html=True)
        try:
            sv, X_in, base_val = model.predict_shap(feature_df)
            sv_flat = sv[0] if sv.ndim == 2 else sv
            st.plotly_chart(shap_waterfall(sv_flat, model.feature_cols, X_in.iloc[0].values, base_val), use_container_width=True)
            pairs = sorted(zip(sv_flat, model.feature_cols, X_in.iloc[0].values), key=lambda x: abs(x[0]), reverse=True)[:6]
            for sv_val, feat_name, feat_val in pairs:
                dc = "#ff3d5a" if sv_val > 0 else "#00e676"
                st.markdown(f"<div style='background:#0d1320;border:1px solid #1a2332;padding:0.4rem 0.7rem;margin:0.15rem 0;position:relative;'><div style='position:absolute;left:0;top:0;bottom:0;width:3px;background:{dc};'></div><div style='display:flex;justify-content:space-between;'><span style='font-family:monospace;font-size:0.75rem;color:#e8f0fa;'>{feat_name}</span><span style='font-family:monospace;font-size:0.78rem;color:{dc};font-weight:600;'>{sv_val:+.4f}</span></div><div style='font-size:0.6rem;color:#5a6d82;'>val={feat_val:.3f}</div></div>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")

        # ── RECOMMENDATIONS ───────────────────────────────────────────────────
        st.markdown("<div class='section-header'>🛡 Mitigation Recommendations</div>", unsafe_allow_html=True)
        recs = []
        row = feature_df.iloc[0]
        if row.get("night_flag", 0): recs.append("🌙 **Nocturnal zone** — deploy flashing signs 20:00–06:00")
        if row.get("ndvi", 0) > 0.6: recs.append("🌲 **Heavy vegetation** — install wildlife detection sensors")
        if row.get("dist_water_km", 99) < 1: recs.append("💧 **Near water** — install crossing structures")
        if row.get("speed_ratio", 0) > 1.1: recs.append("🚗 **Over-speed** — deploy speed cameras")
        if row.get("corridor_dist_km", 99) < 2: recs.append("🗺 **Near corridor** — install wildlife fencing")
        if not recs: recs.append("✅ Risk factors acceptable — standard monitoring")
        for rec in recs: st.markdown(f"- {rec}")

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION C: DATA-RICH INTERACTIVE MAP (always visible)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='section-header'>🗺 Tactical Intelligence Map</div>", unsafe_allow_html=True)

    try:
        pred = st.session_state.get("last_prediction", None)
        map_lat = pred["lat"] if pred else 12.0
        map_lon = pred["lon"] if pred else 76.5
        map_z   = 11 if pred else 6

        tile_map = {"Google Satellite Hybrid": GOOGLE_HYB, "Google Satellite": GOOGLE_SAT, "Google Roads": GOOGLE_ROAD, "ESRI Satellite": ESRI_SAT}
        tile_url = tile_map.get(base_layer)

        if base_layer == "OpenStreetMap (Fast)":
            m = folium.Map(location=[map_lat, map_lon], zoom_start=map_z, tiles="OpenStreetMap", control_scale=True, prefer_canvas=True)
        elif base_layer == "Dark (CartoDB)":
            m = folium.Map(location=[map_lat, map_lon], zoom_start=map_z, tiles="CartoDB dark_matter", control_scale=True, prefer_canvas=True)
        elif tile_url:
            m = folium.Map(location=[map_lat, map_lon], zoom_start=map_z, tiles=None, control_scale=True, prefer_canvas=True)
            subs = GOOGLE_SUBS if "google" in tile_url else None
            folium.TileLayer(tiles=tile_url, attr="Google/ESRI", name=base_layer, max_zoom=20, subdomains=subs or ["a"]).add_to(m)
        else:
            m = folium.Map(location=[map_lat, map_lon], zoom_start=map_z, tiles="OpenStreetMap", control_scale=True, prefer_canvas=True)

        # ── LAYER 1: Risk Heatmap (sampled to max 500 for speed) ─────────────
        if show_heatmap:
            acc = df[df["accident"] == 1][["latitude", "longitude", "risk_score"]].dropna()
            if len(acc) > 500:
                acc = acc.sample(500, random_state=42)
            hd = acc.values.tolist()
            if hd:
                hg = folium.FeatureGroup(name="🌡️ Collision Risk Heatmap")
                HeatMap(hd, min_opacity=0.25, max_zoom=14, radius=20, blur=18,
                        gradient={"0.15":"#06D6A0","0.35":"#FFD166","0.6":"#EF476F","0.85":"#B5179E","1.0":"#ff0040"}).add_to(hg)
                hg.add_to(m)

        # ── LAYER 2: NDVI + Weather (only when toggled, max 30 each) ─────────
        if show_all_data:
            grid = df.groupby([df["latitude"].round(1), df["longitude"].round(1)]).agg(
                ndvi=("ndvi","mean"), risk=("risk_score","mean"),
                acc_rate=("accident","mean"), rain=("rainfall_mm","mean"),
                vis=("visibility_m","mean"), count=("accident","count"),
            ).reset_index().rename(columns={"latitude":"lat","longitude":"lon"})

            ndvi_grp = folium.FeatureGroup(name="🌿 NDVI Vegetation")
            for _, s in grid.head(30).iterrows():
                nv = s.get("ndvi", 0.5)
                nc = "#1b5e20" if nv>0.6 else "#4caf50" if nv>0.45 else "#c8a020" if nv>0.3 else "#8B6914"
                folium.Circle(location=[s["lat"],s["lon"]], radius=5000, color=nc, fill=True, fill_color=nc, fill_opacity=0.2, weight=1,
                    popup=f"🌿 NDVI {nv:.3f}<br>Risk {s.get('risk',0):.3f}", tooltip=f"NDVI {nv:.2f}").add_to(ndvi_grp)
            ndvi_grp.add_to(m)

            wx_grp = folium.FeatureGroup(name="🌧️ Rain Zones")
            for _, s in grid.head(30).iterrows():
                rv = s.get("rain", 0)
                if rv > 15:   wc, wr = "#1565c0", 4000
                elif rv > 5:  wc, wr = "#42a5f5", 2500
                else:         wc, wr = "#90caf9", 1500
                folium.Circle(location=[s["lat"],s["lon"]], radius=wr, color=wc, fill=True, fill_color=wc, fill_opacity=0.12, weight=1,
                    popup=f"🌧 Rain {rv:.1f}mm<br>👁 Vis {s.get('vis',0):.0f}m", tooltip=f"Rain {rv:.1f}mm").add_to(wx_grp)
            wx_grp.add_to(m)

        # ── LAYER 3: Top 20 Risk Segments ────────────────────────────────────
        risk_agg = df.groupby([df["latitude"].round(1), df["longitude"].round(1)]).agg(
            risk=("risk_score","mean"), ndvi=("ndvi","mean"),
            acc_rate=("accident","mean"), count=("accident","count"),
        ).reset_index().rename(columns={"latitude":"lat","longitude":"lon"})
        seg_risk_grp = folium.FeatureGroup(name="⚠️ High Risk Segments")
        for _, s in risk_agg.nlargest(20, "risk").iterrows():
            r = s.get("risk", 0.5)
            sc = "#06D6A0" if r<0.35 else "#FFD166" if r<0.55 else "#EF476F" if r<0.75 else "#B5179E"
            folium.CircleMarker(location=[s["lat"],s["lon"]], radius=4+int(r*7), color=sc, fill=True, fill_color=sc, fill_opacity=0.7,
                popup=f"⚠️ Risk {r:.3f}<br>NDVI {s.get('ndvi',0):.3f}<br>Acc {s.get('acc_rate',0):.1%}", tooltip=f"Risk {r:.2f}").add_to(seg_risk_grp)
        seg_risk_grp.add_to(m)

        # ── LAYER 4: Protected Areas ─────────────────────────────────────────
        if show_pas:
            pg = folium.FeatureGroup(name="🟢 Protected Areas")
            for pa in PROTECTED_AREAS:
                dt = ""
                if pred:
                    d = _haversine(pred["lat"], pred["lon"], pa["lat"], pa["lon"])
                    dt = f"<br>{d:.1f}km away"
                folium.CircleMarker(location=[pa["lat"],pa["lon"]], radius=9, color="#06D6A0", fill=True, fill_color="#06D6A0", fill_opacity=0.8,
                    popup=folium.Popup(f"<b>🟢 {pa['name']}</b><br>{pa.get('state','')}{dt}", max_width=240), tooltip=pa["name"]).add_to(pg)
            pg.add_to(m)

        # ── LAYER 5: Corridors ───────────────────────────────────────────────
        if show_corridors:
            cg = folium.FeatureGroup(name="🟠 Wildlife Corridors")
            for cor in WILDLIFE_CORRIDORS:
                folium.CircleMarker(location=[cor["lat"],cor["lon"]], radius=7, color="#FFD166", fill=True, fill_color="#FFD166", fill_opacity=0.8,
                    popup=folium.Popup(f"<b>🟠 {cor['name']}</b>", max_width=240), tooltip=cor["name"]).add_to(cg)
            cg.add_to(m)

        # ── LAYER 6: Pipeline prediction data ────────────────────────────────
        if pred:
            feat = pred["features"]
            spatial = pred.get("spatial") or {}
            loc_name = pred.get("location_name", "Location")
            risk_score = feat.get("driver_risk", 0.5)
            rc = "#06D6A0" if risk_score<0.4 else "#FFD166" if risk_score<0.7 else "#EF476F" if risk_score<1.0 else "#B5179E"

            popup_html = (
                f"<div style='font-family:sans-serif;min-width:260px;'>"
                f"<h3 style='margin:0 0 6px;color:{rc};'>📍 {loc_name}</h3>"
                f"<table style='font-size:11px;width:100%;'>"
                f"<tr><td><b>🦁 Species</b></td><td>{feat.get('species','—').capitalize()} ({feat.get('species_risk',0):.0%})</td></tr>"
                f"<tr><td><b>🌡️ Temp</b></td><td>{feat.get('temperature_c',0):.1f}°C · {feat.get('humidity_pct',0):.0f}%</td></tr>"
                f"<tr><td><b>🌧️ Rain</b></td><td>{feat.get('rainfall_mm',0):.1f}mm</td></tr>"
                f"<tr><td><b>👁 Vis</b></td><td>{feat.get('visibility_m',0)}m</td></tr>"
                f"<tr><td><b>🌿 NDVI</b></td><td>{feat.get('ndvi',0):.3f}</td></tr>"
                f"<tr><td><b>🏞️ PA</b></td><td>{spatial.get('nearest_pa','—')} ({spatial.get('protected_dist_km',0):.1f}km)</td></tr>"
                f"<tr><td><b>🚗 Speed</b></td><td>{feat.get('actual_speed',0):.0f}/{feat.get('speed_limit',0)} km/h</td></tr>"
                f"<tr><td><b>⚠️ Risk</b></td><td style='color:{rc};font-weight:bold;'>{risk_score:.3f}</td></tr>"
                f"</table></div>"
            )

            pred_grp = folium.FeatureGroup(name="📍 Prediction")
            folium.Marker(location=[pred["lat"],pred["lon"]], popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"📍 {loc_name} — Risk {risk_score:.3f}", icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa")).add_to(pred_grp)
            folium.Circle(location=[pred["lat"],pred["lon"]], radius=5000, color=rc, fill=True, fill_opacity=0.1, weight=2).add_to(pred_grp)
            HeatMap([[pred["lat"],pred["lon"],risk_score]], min_opacity=0.5, radius=45, blur=30,
                    gradient={"0.2":"#06D6A0","0.5":"#FFD166","0.75":"#EF476F","1.0":"#B5179E"}).add_to(pred_grp)

            # Weather badge
            temp_v = feat.get("temperature_c", 0)
            rain_v = feat.get("rainfall_mm", 0)
            vis_v  = feat.get("visibility_m", 0)
            folium.Marker(location=[pred["lat"]+0.015, pred["lon"]+0.015],
                icon=folium.DivIcon(html=f"<div style='background:#1a237e;color:white;border-radius:8px;padding:2px 5px;font-size:9px;font-family:monospace;border:1px solid #42a5f5;white-space:nowrap;'>🌡{temp_v:.0f}°C 🌧{rain_v:.0f}mm 👁{vis_v:.0f}m</div>", icon_size=(150,22))).add_to(pred_grp)

            # NDVI ring
            ndvi_val = feat.get("ndvi", 0.5)
            ndvi_c = "#1b5e20" if ndvi_val>0.6 else "#4caf50" if ndvi_val>0.45 else "#c8a020" if ndvi_val>0.3 else "#8B6914"
            folium.Circle(location=[pred["lat"],pred["lon"]], radius=3000, color=ndvi_c, fill=True, fill_color=ndvi_c, fill_opacity=0.12, weight=1).add_to(pred_grp)
            pred_grp.add_to(m)

            # Sub-locations
            sub_locs = pred.get("sub_locations", [])
            if sub_locs:
                sg = folium.FeatureGroup(name=f"🎯 Sub-Locations ({len(sub_locs)})")
                for i, sl in enumerate(sub_locs[:20]):
                    folium.CircleMarker(location=[sl["lat"],sl["lon"]], radius=6, color="#00e5ff", fill=True, fill_color="#00e5ff", fill_opacity=0.6,
                        popup=f"🎯 {sl['name'][:60]}", tooltip=f"#{i+1}").add_to(sg)
                sg.add_to(m)

            # Pipeline source badges
            log_data = pred.get("log", [])
            for li, l in enumerate(log_data[:4]):
                src_type = l.get('source_type','')
                bg = {"api":"#00695c","news":"#e65100","government":"#4a148c","computation":"#1565c0"}.get(src_type, "#333")
                emoji = {"api":"🌐","news":"📰","government":"🏛️","computation":"⚙️"}.get(src_type, "📡")
                offset_lat = 0.025 * (1 if li < 2 else -1)
                offset_lon = 0.025 * (1 if li % 2 == 0 else -1)
                folium.Marker(
                    location=[pred["lat"]+offset_lat, pred["lon"]+offset_lon],
                    icon=folium.DivIcon(html=f"<div style='background:{bg};color:white;border-radius:5px;padding:1px 4px;font-size:8px;font-family:monospace;white-space:nowrap;'>{emoji} {l.get('source','')[:18]} {'✅' if l['status']=='success' else '⚠️'}</div>", icon_size=(140,16))
                ).add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)

        # ── Render Map + Sidebar ─────────────────────────────────────────────
        cl, cr = st.columns([3, 1])
        with cl:
            st_folium(m, width=None, height=600, returned_objects=[])
        with cr:
            if pred:
                st.markdown("<div class='section-header'>📍 Extracted Intel</div>", unsafe_allow_html=True)
                st.markdown(f"""
                <div class='metric-card'><h3>📍 Location</h3><div class='value' style='font-size:0.82rem;'>{loc_name[:35]}</div><div class='sub'>{pred['lat']:.4f}, {pred['lon']:.4f}</div></div>
                <div class='metric-card'><h3>🌡️ Weather</h3><div class='value' style='font-size:0.9rem;'>{feat.get('temperature_c',0):.1f}°C · {feat.get('humidity_pct',0):.0f}%</div><div class='sub'>Rain {feat.get('rainfall_mm',0):.1f}mm · Vis {feat.get('visibility_m',0)}m</div></div>
                <div class='metric-card'><h3>🌿 NDVI</h3><div class='value' style='font-size:1.1rem; color:{ndvi_c};'>{feat.get('ndvi',0):.3f}</div><div class='sub'>{'Dense' if ndvi_val>0.6 else 'Good' if ndvi_val>0.45 else 'Moderate' if ndvi_val>0.3 else 'Sparse'}</div></div>
                <div class='metric-card'><h3>🦁 Species</h3><div class='value' style='font-size:0.9rem;'>{feat.get('species','—').capitalize()}</div><div class='sub'>Risk: {feat.get('species_risk',0):.0%}</div></div>
                <div class='metric-card'><h3>🏞️ PA</h3><div class='value' style='font-size:0.82rem;'>{spatial.get('nearest_pa','—')}</div><div class='sub'>{spatial.get('protected_dist_km',0):.1f}km</div></div>
                <div class='metric-card'><h3>🚗 Speed</h3><div class='value' style='font-size:0.9rem;'>{feat.get('actual_speed',0):.0f}/{feat.get('speed_limit',0):.0f}</div><div class='sub'>Ratio {feat.get('speed_ratio',0):.2f}</div></div>
                """, unsafe_allow_html=True)
                st.markdown("<div class='section-header'>📡 Pipeline</div>", unsafe_allow_html=True)
                summary = pred["summary"]
                st.markdown(f"**{summary['success_count']}/{summary['total_sources']}** · **{summary['total_time_ms']}ms**")
                for l in pred["log"]:
                    ic = "✅" if l["status"]=="success" else "⚠️"
                    st.markdown(f"<div style='font-size:0.65rem;'>{ic} {l['source'][:22]} · {l['response_ms']}ms</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='section-header'>Map Layers</div>", unsafe_allow_html=True)
                legends = [("#EF476F","🌡️ Risk Heatmap"),("#1b5e20","🌿 NDVI Dense"),("#4caf50","🌿 NDVI Good"),("#1565c0","🌧️ Heavy Rain"),("#06D6A0","🟢 Protected Areas"),("#FFD166","🟠 Corridors"),("#00e5ff","🎯 Sub-Locations")]
                for color, label in legends:
                    st.markdown(f"<div style='display:flex;align-items:center;gap:0.4rem;margin:0.15rem 0;'><div style='width:11px;height:11px;border-radius:50%;background:{color};'></div><span style='font-size:0.7rem;'>{label}</span></div>", unsafe_allow_html=True)
                st.info("🔎 Search → Extract → Predict to populate map with live data")

    except Exception as map_err:
        st.error(f"Map rendering error: {map_err}")
        # Fallback: always show a basic map
        m_fallback = folium.Map(location=[12.0, 76.5], zoom_start=6, tiles="OpenStreetMap")
        for pa in PROTECTED_AREAS:
            folium.CircleMarker(location=[pa["lat"],pa["lon"]], radius=7, color="#06D6A0", fill=True, tooltip=pa["name"]).add_to(m_fallback)
        st_folium(m_fallback, width=None, height=500, returned_objects=[])


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
                    marker=dict(size=12, color=color, opacity=0.85, line=dict(width=1, color='#1a2332')),
                    text=subset['highway_segment'].str[:18],
                    textposition='top center',
                    textfont=dict(size=8, color=color),
                    name=cat,
                    hovertemplate='<b>%{text}</b><br>Accident Rate: %{x:.1%}<br>Hotspot Score: %{y:.3f}<extra></extra>',
                ))
        fig_scatter.add_vline(x=median_acc, line_dash="dash", line_color="#5a6d82", opacity=0.5)
        fig_scatter.add_hline(y=median_risk, line_dash="dash", line_color="#5a6d82", opacity=0.5)
        fig_scatter.update_layout(
            xaxis_title="Historical Accident Rate", yaxis_title="Hotspot Score (Predicted Risk)",
            template="plotly_dark", paper_bgcolor="#060a10", plot_bgcolor="#0d1320",
            font=dict(color="#c8d6e5", family="'JetBrains Mono', monospace"), height=500,
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # ── SHAP Global Feature Importance ───────────────────────────────────
        st.markdown("<div class='section-header'>🧠 SHAP — Global Feature Importance (What Drives Risk Overall)</div>", unsafe_allow_html=True)
        st.markdown("""
        <p style='color:#5a6d82; font-size:0.72rem; font-family:"IBM Plex Mono",monospace;'>
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
                    colorscale=[[0, '#00e676'], [0.5, '#ffb020'], [1.0, '#ff3d5a']],
                    line=dict(width=1, color='rgba(255,255,255,0.1)'),
                ),
                hovertemplate='<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>',
            ))
            fig_shap.update_layout(
                template="plotly_dark", paper_bgcolor="#060a10", plot_bgcolor="#0d1320",
                font=dict(color="#c8d6e5", family="'JetBrains Mono', monospace"), height=450,
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
