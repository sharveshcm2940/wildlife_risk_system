"""
Synthetic Wildlife-Vehicle Collision Data Generator
Generates realistic training data for the WVC Risk Prediction System
Covers both South India (Western Ghats) and Central India wildlife corridors
with highway segment-level resolution
"""

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

np.random.seed(42)

# ── Geographic bounds (India — full wildlife corridor coverage) ──────────────
LAT_MIN, LAT_MAX = 8.5, 24.5
LON_MIN, LON_MAX = 74.0, 82.0

# ── Highway segments (real corridors through forest reserves) ────────────────
# South India — Western Ghats highways that intersect forest reserves
SOUTH_INDIA_HIGHWAY_SEGMENTS = [
    # NH-766 Mysore-Wayanad (Bandipur corridor)
    {"name": "NH-766 Bandipur-Wayanad", "lat": 11.66, "lon": 76.63, "state": "Karnataka", "risk_base": 0.85},
    {"name": "NH-766 Gundlupet Stretch", "lat": 11.78, "lon": 76.67, "state": "Karnataka", "risk_base": 0.80},
    {"name": "NH-766 Sultan Bathery", "lat": 11.66, "lon": 76.24, "state": "Kerala", "risk_base": 0.75},
    # NH-181 Mysore-Ooty (Mudumalai corridor)
    {"name": "NH-181 Mudumalai Stretch", "lat": 11.56, "lon": 76.55, "state": "Tamil Nadu", "risk_base": 0.88},
    {"name": "NH-181 Theppakadu", "lat": 11.53, "lon": 76.53, "state": "Tamil Nadu", "risk_base": 0.82},
    {"name": "NH-181 Masinagudi", "lat": 11.57, "lon": 76.65, "state": "Tamil Nadu", "risk_base": 0.78},
    # Nagarhole / Rajiv Gandhi NP
    {"name": "Nagarhole SH-33", "lat": 12.05, "lon": 76.15, "state": "Karnataka", "risk_base": 0.80},
    {"name": "Hunsur-Nagarhole Road", "lat": 12.10, "lon": 76.20, "state": "Karnataka", "risk_base": 0.72},
    # Sathyamangalam corridor (Tamil Nadu)
    {"name": "NH-948 Sathyamangalam", "lat": 11.50, "lon": 77.25, "state": "Tamil Nadu", "risk_base": 0.76},
    {"name": "Hasanur-Talavadi Road", "lat": 11.60, "lon": 77.05, "state": "Tamil Nadu", "risk_base": 0.70},
    # Periyar (Kerala)
    {"name": "NH-183 Kumily-Periyar", "lat": 9.47, "lon": 77.17, "state": "Kerala", "risk_base": 0.72},
    {"name": "Thekkady Forest Road", "lat": 9.50, "lon": 77.13, "state": "Kerala", "risk_base": 0.68},
    # BR Hills (Karnataka)
    {"name": "BR Hills Forest Road", "lat": 11.99, "lon": 77.16, "state": "Karnataka", "risk_base": 0.73},
    {"name": "Chamarajanagar-BR Hills", "lat": 11.92, "lon": 77.05, "state": "Karnataka", "risk_base": 0.65},
    # Coorg / Talacauvery
    {"name": "Madikeri-Talacauvery", "lat": 12.38, "lon": 75.50, "state": "Karnataka", "risk_base": 0.58},
    # Anamalai / Parambikulam (Tamil Nadu / Kerala border)
    {"name": "NH-17 Anamalai Hills", "lat": 10.50, "lon": 76.95, "state": "Tamil Nadu", "risk_base": 0.70},
    {"name": "Parambikulam Road", "lat": 10.42, "lon": 76.80, "state": "Kerala", "risk_base": 0.65},
    # Nilgiri Biosphere
    {"name": "NH-181 Gudalur Section", "lat": 11.50, "lon": 76.50, "state": "Tamil Nadu", "risk_base": 0.77},
    {"name": "Kotagiri Forest Road", "lat": 11.42, "lon": 76.85, "state": "Tamil Nadu", "risk_base": 0.60},
    # Agasthyamalai (Southernmost)
    {"name": "Agasthyamalai Route", "lat": 8.65, "lon": 77.23, "state": "Tamil Nadu", "risk_base": 0.55},
]

# Central India — Tiger corridors
CENTRAL_INDIA_HIGHWAY_SEGMENTS = [
    {"name": "NH-7 Kanha-Pench Corridor", "lat": 22.33, "lon": 80.62, "state": "Madhya Pradesh", "risk_base": 0.82},
    {"name": "NH-44 Pench Crossing", "lat": 21.72, "lon": 79.30, "state": "Madhya Pradesh", "risk_base": 0.78},
    {"name": "Tadoba NH-6 Stretch", "lat": 20.20, "lon": 79.35, "state": "Maharashtra", "risk_base": 0.75},
    {"name": "Satpura Forest Road", "lat": 22.52, "lon": 78.12, "state": "Madhya Pradesh", "risk_base": 0.70},
    {"name": "Melghat NH-6", "lat": 21.40, "lon": 77.00, "state": "Maharashtra", "risk_base": 0.72},
    {"name": "Nagzira-Navegaon Corr.", "lat": 21.10, "lon": 79.50, "state": "Maharashtra", "risk_base": 0.65},
    {"name": "Panna NH-75 Segment", "lat": 24.72, "lon": 80.55, "state": "Madhya Pradesh", "risk_base": 0.62},
    {"name": "Ranthambore AH-48", "lat": 26.02, "lon": 76.45, "state": "Rajasthan", "risk_base": 0.68},
]

ALL_HIGHWAY_SEGMENTS = SOUTH_INDIA_HIGHWAY_SEGMENTS + CENTRAL_INDIA_HIGHWAY_SEGMENTS

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
    """Generate coordinates clustered around real highway segments through forests.
    70% of data from South India, 30% from Central India."""
    lats, lons = [], []

    # 70% near South India highway segments
    n_south = int(n * 0.70)
    per_seg = n_south // len(SOUTH_INDIA_HIGHWAY_SEGMENTS)
    for seg in SOUTH_INDIA_HIGHWAY_SEGMENTS:
        # Tight clustering around actual highway coords (sigma ~0.15° ≈ 15km)
        lats.extend(np.random.normal(seg["lat"], 0.15, per_seg))
        lons.extend(np.random.normal(seg["lon"], 0.15, per_seg))

    # 30% near Central India
    n_central = n - n_south
    per_seg_c = n_central // len(CENTRAL_INDIA_HIGHWAY_SEGMENTS)
    for seg in CENTRAL_INDIA_HIGHWAY_SEGMENTS:
        lats.extend(np.random.normal(seg["lat"], 0.20, per_seg_c))
        lons.extend(np.random.normal(seg["lon"], 0.20, per_seg_c))

    # Fill remainder with random points within bounds
    remainder = n - len(lats)
    if remainder > 0:
        lats.extend(np.random.uniform(LAT_MIN, LAT_MAX, remainder))
        lons.extend(np.random.uniform(LON_MIN, LON_MAX, remainder))

    lats = np.clip(lats[:n], LAT_MIN, LAT_MAX)
    lons = np.clip(lons[:n], LON_MIN, LON_MAX)
    return np.array(lats), np.array(lons)


def assign_highway_segment(lat: float, lon: float) -> str:
    """Find nearest highway segment for each data point."""
    best_seg, best_dist = "Unknown", 999
    for seg in ALL_HIGHWAY_SEGMENTS:
        dist = ((lat - seg["lat"])**2 + (lon - seg["lon"])**2)**0.5
        if dist < best_dist:
            best_dist = dist
            best_seg = seg["name"]
    return best_seg


def generate_dataset(n: int = 12_000) -> pd.DataFrame:
    lats, lons = generate_coords(n)

    hour        = np.random.randint(0, 24, n)
    day_of_week = np.random.randint(0, 7, n)
    season      = np.random.choice(SEASONS, n)
    road_type   = np.random.choice(ROAD_TYPES, n)
    species     = np.random.choice(SPECIES, n)

    # Assign highway segments
    segments = [assign_highway_segment(lats[i], lons[i]) for i in range(n)]

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
        'highway_segment':   segments,
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
    print(f"South India records: {(df['latitude'] < 15).sum():,}")
    print(f"Central India records: {(df['latitude'] >= 15).sum():,}")
    print(f"Unique segments: {df['highway_segment'].nunique()}")
    print(df.describe().T[['mean', 'std', 'min', 'max']].to_string())
