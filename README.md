# 🐾 WildGuard AI — Wildlife-Vehicle Collision Risk Prediction System

A production-grade ML system for predicting wildlife-vehicle collision risk using XGBoost, SHAP explainability, and a Streamlit geospatial dashboard.

---

## 🏗 Architecture

```
wildlife_risk/
├── app.py                  # Main Streamlit application (5 pages)
├── requirements.txt        # All dependencies
├── data/
│   └── generate_data.py    # Synthetic dataset generator (12,000 records)
├── models/
│   ├── train.py            # XGBoost training pipeline + SHAP
│   └── [auto-generated]    # xgb_model.pkl, encoders.pkl, metrics.json ...
└── utils/
    └── helpers.py          # Folium maps, Plotly charts, risk scoring
```

---

## 🚀 Quick Start

```bash
cd wildlife_risk
pip install -r requirements.txt
streamlit run app.py
```

The model trains automatically on first launch (~60 seconds).

---

## 🌟 Feature Set

### Core Features (High Weight)
| Feature | Description |
|---|---|
| 📍 Latitude / Longitude | Spatial anchor for all GIS logic |
| 🌲 NDVI | Satellite forest density index |
| 💧 Distance to water | Wildlife convergence proxy |
| 🚗 Speed limit + actual speed | Real vs posted speed |
| 🌙 Hour of day | Dawn/dusk peak crossing |
| 🗺 Road type | Highway / rural / forest road |
| 🌧 Rainfall + visibility | Driver perception impairment |
| 📉 Past accident count | Historical hotspot density |

### 🏆 Advanced Engineered Features
| Feature | How It's Built |
|---|---|
| **Animal Movement Score** | Composite: NDVI × water × dawn/dusk × breeding × night light |
| **KDE Accident Hotspot** | Kernel density over past accident spatial distribution |
| **Driver Risk Index** | speed_ratio × night_penalty × road_type_risk |
| **Wildlife Corridor Proximity** | GIS-derived corridor distance |
| **Breeding Season Flag** | Monsoon + post-monsoon encoding |
| **Species-Specific Risk** | Per-animal behavior encoding (tiger=0.9 … nilgai=0.5) |
| **Nighttime Light Intensity** | Satellite-derived human activity proxy (0–255) |
| **Rolling 7-Day Trend** | Temporal autocorrelation signal |

### Standard Features
Temporal: day_of_week, season, rush_hour  
Road geometry: curvature_deg_km, street_lighting, road_width_m  
Environmental: protected_dist_km, temperature_c, humidity_pct

---

## 🤖 Model Details

| Parameter | Value |
|---|---|
| Algorithm | XGBoost (`XGBClassifier`) |
| n_estimators | 500 (with early stopping @30) |
| max_depth | 6 |
| learning_rate | 0.05 |
| class_weight | auto-balanced (`scale_pos_weight`) |
| Evaluation | ROC-AUC, Avg Precision, Brier Score |
| Explainability | TreeSHAP (global + per-prediction) |

---

## 📱 Application Pages

1. **🏠 Dashboard** — KPI cards, hourly risk, species heatmaps, confusion matrix
2. **🔮 Risk Predictor** — Real-time prediction with 24 input features + SHAP waterfall
3. **🗺 Risk Map** — Folium heatmap with KDE overlay + filterable markers
4. **📊 Analytics** — Deep-dive tabs: temporal, environmental, traffic, wildlife, historical
5. **🧠 Model Insights** — Performance metrics, SHAP importance, beeswarm, feature groups

---

## 📊 Expected Performance (on 12K synthetic records)
- ROC-AUC: ~0.88–0.92
- Avg Precision: ~0.85–0.90
- Accuracy: ~0.82–0.86

---

## 🗺 Geographic Scope
Default data covers **Central India / Western Ghats** wildlife corridors  
(lat 18.5–24.5°N, lon 74–82°E). Easily replaceable with real GPS data.

---

## 🔄 Using Real Data

Replace `data/generate_data.py` with your actual dataset loader.  
Ensure your CSV/database has these columns (or a subset):
- `latitude`, `longitude`, `accident` (0/1), `risk_score` (0–1)
- All feature columns listed in the Feature Set above

Then retrain:
```python
from models.train import WildlifeRiskModel
model = WildlifeRiskModel()
model.train(your_dataframe)
model.save()
```
