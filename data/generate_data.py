"""
Synthetic Wildlife-Vehicle Collision Data Generator
Generates realistic training data for the WVC Risk Prediction System
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

np.random.seed(42)

# ── Geographic bounds (India – Western Ghats / Central India wildlife corridors) ──
LAT_MIN, LAT_MAX = 18.5, 24.5
LON_MIN, LON_MAX = 74.0, 82.0

# Wildlife hotspot centroids
HOTSPOTS = [
    (20.9, 76.0), (21.5, 80.5), (23.1, 78.8),
    (19.8, 77.2), (22.3, 75.5), (23.8, 81.0),
    (20.2, 79.5), (21.0, 77.8),
]

ROAD_TYPES = ['highway', 'rural', 'forest_road', 'state_highway', 'national_highway']
SPECIES     = ['tiger', 'leopard', 'elephant', 'deer', 'boar', 'wolf', 'nilgai', 'sambar']
SEASONS     = ['summer', 'monsoon', 'post_monsoon', 'winter']

SPECIES_RISK = {'tiger': 0.9, 'elephant': 0.95, 'leopard': 0.85,
                'deer': 0.60, 'boar': 0.55, 'wolf': 0.70,
                'nilgai': 0.50, 'sambar': 0.65}

ROAD_RISK   = {'forest_road': 0.90, 'rural': 0.65, 'state_highway': 0.55,
               'highway': 0.45, 'national_highway': 0.40}

SEASON_RISK = {'monsoon': 0.90, 'post_monsoon': 0.70,
               'winter': 0.55,  'summer': 0.40}


def generate_coords(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Cluster coords around known hotspots with spatial noise."""
    lats, lons = [], []
    per_hotspot = n // len(HOTSPOTS)
    for (hlat, hlon) in HOTSPOTS:
        lats.extend(np.random.normal(hlat, 0.8, per_hotspot))
        lons.extend(np.random.normal(hlon, 0.8, per_hotspot))
    remainder = n - len(lats)
    lats.extend(np.random.uniform(LAT_MIN, LAT_MAX, remainder))
    lons.extend(np.random.uniform(LON_MIN, LON_MAX, remainder))
    lats = np.clip(lats, LAT_MIN, LAT_MAX)
    lons = np.clip(lons, LON_MIN, LON_MAX)
    return np.array(lats[:n]), np.array(lons[:n])


def generate_dataset(n: int = 12_000) -> pd.DataFrame:
    lats, lons = generate_coords(n)

    hour        = np.random.randint(0, 24, n)
    day_of_week = np.random.randint(0, 7, n)
    season      = np.random.choice(SEASONS, n)
    road_type   = np.random.choice(ROAD_TYPES, n)
    species     = np.random.choice(SPECIES, n)

    # ── Core features ────────────────────────────────────────────────────────────
    ndvi            = np.clip(np.random.normal(0.55, 0.2, n), 0.0, 1.0)
    dist_water      = np.abs(np.random.exponential(2.5, n))           # km
    speed_limit     = np.random.choice([30, 40, 60, 80, 100], n)
    actual_speed    = speed_limit * np.random.uniform(0.7, 1.4, n)
    rainfall        = np.random.exponential(8, n)                     # mm
    visibility      = np.clip(1000 - rainfall * 15 + np.random.normal(0, 80, n), 50, 1000)  # m
    past_accidents  = np.random.negative_binomial(2, 0.3, n)

    # ── Advanced / engineered features ───────────────────────────────────────────
    night_flag      = ((hour < 6) | (hour >= 20)).astype(int)
    dawn_dusk       = ((hour >= 5) & (hour <= 7)) | ((hour >= 17) & (hour <= 19))
    dawn_dusk       = dawn_dusk.astype(int)
    rush_hour       = (((hour >= 6) & (hour <= 9)) | ((hour >= 17) & (hour <= 20))).astype(int)

    breeding_season = np.where(np.isin(season, ['monsoon', 'post_monsoon']), 1, 0)
    species_risk_val= np.array([SPECIES_RISK[s] for s in species])
    road_risk_val   = np.array([ROAD_RISK[r]    for r in road_type])
    season_risk_val = np.array([SEASON_RISK[s]  for s in season])

    # Speed ratio
    speed_ratio     = actual_speed / speed_limit

    # Driver Risk Index: speed_ratio × night_penalty × road_type_risk
    driver_risk     = speed_ratio * (1 + 0.6 * night_flag) * road_risk_val

    # Wildlife corridor proximity (synthetic GIS proxy)
    corridor_dist   = np.clip(
        np.abs(np.random.normal(1.5, 1.2, n)) + 0.3 * (1 - ndvi), 0, 10
    )

    # Nighttime light intensity (0-255 scale, inverse of wildlife presence)
    night_light     = np.clip(
        np.random.normal(30, 20, n) + 15 * (road_risk_val < 0.6).astype(float),
        0, 255
    )

    # Road curvature (degrees per km)
    curvature       = np.abs(np.random.normal(12, 8, n))
    street_lighting = np.random.binomial(1, 0.35, n)
    road_width      = np.random.choice([4, 6, 7, 10, 14], n)  # metres
    protected_dist  = np.abs(np.random.exponential(5, n))      # km to nearest PA
    temperature     = np.random.normal(28, 8, n)               # °C
    humidity        = np.clip(np.random.normal(65, 20, n), 20, 100)

    # Rolling 7-day trend (synthetic autocorrelation signal)
    rolling_7day    = past_accidents * np.random.uniform(0.8, 1.3, n)

    # Animal Movement Score (composite)
    movement_score  = (
        0.30 * ndvi +
        0.25 * np.clip(1 / (dist_water + 0.1), 0, 1) +
        0.20 * dawn_dusk +
        0.15 * breeding_season +
        0.10 * (1 - night_light / 255)
    )

    # KDE accident hotspot density (proxy)
    kde_density     = past_accidents / (corridor_dist + 0.5) * 0.4

    # ── Target variable ───────────────────────────────────────────────────────────
    risk_score = (
        0.18 * movement_score +
        0.14 * driver_risk / 3 +
        0.12 * road_risk_val +
        0.10 * species_risk_val +
        0.08 * season_risk_val +
        0.08 * night_flag * 0.8 +
        0.07 * (rainfall / 50).clip(0, 1) +
        0.07 * (past_accidents / 20).clip(0, 1) +
        0.05 * breeding_season +
        0.05 * (1 - visibility / 1000) +
        0.03 * (curvature / 50).clip(0, 1) +
        0.03 * (1 - street_lighting) * 0.5
    )
    noise     = np.random.normal(0, 0.04, n)
    risk_score = np.clip(risk_score + noise, 0, 1)
    accident   = (risk_score > np.random.uniform(0.4, 0.75, n)).astype(int)

    df = pd.DataFrame({
        # Spatial
        'latitude':          lats,
        'longitude':         lons,
        # Core
        'ndvi':              ndvi,
        'dist_water_km':     dist_water,
        'speed_limit':       speed_limit,
        'actual_speed':      actual_speed.round(1),
        'hour':              hour,
        'road_type':         road_type,
        'rainfall_mm':       rainfall.round(1),
        'visibility_m':      visibility.round(0).astype(int),
        'past_accidents':    past_accidents,
        # Temporal
        'day_of_week':       day_of_week,
        'season':            season,
        'rush_hour':         rush_hour,
        # Advanced
        'movement_score':    movement_score.round(4),
        'kde_density':       kde_density.round(4),
        'driver_risk':       driver_risk.round(4),
        'corridor_dist_km':  corridor_dist.round(3),
        'breeding_season':   breeding_season,
        'species':           species,
        'species_risk':      species_risk_val,
        'night_light':       night_light.round(1),
        'rolling_7day':      rolling_7day.round(2),
        # Road geometry
        'curvature_deg_km':  curvature.round(2),
        'street_lighting':   street_lighting,
        'road_width_m':      road_width,
        # Environmental
        'protected_dist_km': protected_dist.round(3),
        'temperature_c':     temperature.round(1),
        'humidity_pct':      humidity.round(1),
        # Derived
        'night_flag':        night_flag,
        'dawn_dusk':         dawn_dusk,
        'speed_ratio':       speed_ratio.round(3),
        # Target
        'risk_score':        risk_score.round(4),
        'accident':          accident,
    })

    return df


if __name__ == "__main__":
    df = generate_dataset(12_000)
    df.to_csv("wildlife_accidents.csv", index=False)
    print(f"Generated {len(df):,} records  |  Accident rate: {df['accident'].mean():.1%}")
    print(df.describe().T[['mean', 'std', 'min', 'max']].to_string())
