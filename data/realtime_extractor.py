"""
Real-Time Data Extraction Pipeline v3.0
Fetches live data from APIs, news websites, and Indian government portals.

Sources:
  1. Open-Meteo API        — Weather data
  2. Overpass API           — Road + water from OpenStreetMap
  3. GBIF API               — Wildlife occurrence data
  4. System Clock           — Temporal features
  5. GIS Computation        — NDVI, corridor/PA distance
  6. News Websites (RSS)    — The Hindu, NDTV, Down to Earth, Google News
  7. Govt of India Portals  — NTCA, State Forest Depts, MoEFCC, India Biodiversity Portal
"""

import requests
import numpy as np
import pandas as pd
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
from typing import Optional
import time

# ── South India + Central India Protected Areas ──────────────────────────────
PROTECTED_AREAS = [
    # ─── South India (Primary Focus) ───
    {"name": "Bandipur National Park",          "lat": 11.66, "lon": 76.63, "state": "Karnataka"},
    {"name": "Nagarhole (Rajiv Gandhi) NP",     "lat": 12.05, "lon": 76.15, "state": "Karnataka"},
    {"name": "BR Hills Wildlife Sanctuary",     "lat": 11.99, "lon": 77.16, "state": "Karnataka"},
    {"name": "Bhadra Tiger Reserve",            "lat": 13.70, "lon": 75.64, "state": "Karnataka"},
    {"name": "Dandeli-Anshi Tiger Reserve",     "lat": 15.25, "lon": 74.58, "state": "Karnataka"},
    {"name": "Mudumalai Tiger Reserve",         "lat": 11.56, "lon": 76.55, "state": "Tamil Nadu"},
    {"name": "Sathyamangalam Tiger Reserve",    "lat": 11.50, "lon": 77.25, "state": "Tamil Nadu"},
    {"name": "Anamalai Tiger Reserve",          "lat": 10.40, "lon": 76.85, "state": "Tamil Nadu"},
    {"name": "Kalakkad-Mundanthurai TR",        "lat": 8.60,  "lon": 77.35, "state": "Tamil Nadu"},
    {"name": "Periyar Tiger Reserve",           "lat": 9.47,  "lon": 77.17, "state": "Kerala"},
    {"name": "Wayanad Wildlife Sanctuary",      "lat": 11.60, "lon": 76.02, "state": "Kerala"},
    {"name": "Parambikulam Tiger Reserve",      "lat": 10.42, "lon": 76.83, "state": "Kerala"},
    {"name": "Silent Valley National Park",     "lat": 11.08, "lon": 76.42, "state": "Kerala"},
    {"name": "Nagarjunasagar-Srisailam TR",     "lat": 15.85, "lon": 78.87, "state": "Andhra Pradesh"},
    # ─── Central India ───
    {"name": "Panna Tiger Reserve",             "lat": 24.72, "lon": 80.02, "state": "Madhya Pradesh"},
    {"name": "Kanha National Park",             "lat": 22.33, "lon": 80.62, "state": "Madhya Pradesh"},
    {"name": "Tadoba-Andhari Reserve",          "lat": 20.20, "lon": 79.35, "state": "Maharashtra"},
    {"name": "Satpura Tiger Reserve",           "lat": 22.52, "lon": 78.12, "state": "Madhya Pradesh"},
    {"name": "Melghat Tiger Reserve",           "lat": 21.42, "lon": 76.30, "state": "Maharashtra"},
    {"name": "Pench Tiger Reserve",             "lat": 21.72, "lon": 79.30, "state": "Madhya Pradesh"},
    {"name": "Bandhavgarh National Park",       "lat": 23.72, "lon": 80.95, "state": "Madhya Pradesh"},
]

WILDLIFE_CORRIDORS = [
    # ─── South India (Nilgiri Biosphere + corridors) ───
    {"name": "Bandipur-Mudumalai Corridor",          "lat": 11.61, "lon": 76.59},
    {"name": "Nagarhole-Wayanad Corridor",           "lat": 11.82, "lon": 76.08},
    {"name": "BR Hills-Sathyamangalam Corridor",     "lat": 11.75, "lon": 77.20},
    {"name": "Periyar-Anamalai Corridor",            "lat": 9.95,  "lon": 77.00},
    {"name": "Sathyamangalam-Cauvery Corridor",      "lat": 11.85, "lon": 77.50},
    {"name": "Parambikulam-Anamalai Corridor",       "lat": 10.41, "lon": 76.84},
    {"name": "Dandeli-Anshi-Goa Corridor",           "lat": 15.40, "lon": 74.50},
    # ─── Central India ───
    {"name": "Kanha-Pench Corridor",                 "lat": 22.00, "lon": 80.10},
    {"name": "Tadoba-Navegaon Corridor",             "lat": 20.65, "lon": 79.65},
    {"name": "Satpura-Melghat Corridor",             "lat": 21.95, "lon": 77.20},
    {"name": "Panna-Gangau Corridor",                "lat": 24.60, "lon": 80.20},
]

# ── Indian Government & News Source URLs ─────────────────────────────────────
NEWS_RSS_FEEDS = {
    "The Hindu — Environment": "https://www.thehindu.com/sci-tech/energy-and-environment/feeder/default.rss",
    "Down to Earth — Wildlife": "https://www.downtoearth.org.in/rss/wildlife",
    "NDTV — Environment": "https://feeds.feedburner.com/ndtv/environment-news",
    "Google News — Wildlife Road India": "https://news.google.com/rss/search?q=wildlife+animal+road+accident+India&hl=en-IN&gl=IN&ceid=IN:en",
}

GOVT_SOURCES = {
    "NTCA — Tiger Conservation": {
        "url": "https://ntca.gov.in",
        "desc": "National Tiger Conservation Authority — tiger reserve monitoring, corridor status",
    },
    "Karnataka Forest Dept": {
        "url": "https://aranya.gov.in",
        "desc": "Karnataka State Forest Department — wildlife alerts, Bandipur/Nagarhole updates",
    },
    "Kerala Forest Dept": {
        "url": "https://forest.kerala.gov.in",
        "desc": "Kerala Forest & Wildlife Dept — Periyar/Wayanad elephant corridor data",
    },
    "TN Forest Dept": {
        "url": "https://forests.tn.gov.in",
        "desc": "Tamil Nadu Forest Dept — Mudumalai/Sathyamangalam wildlife data",
    },
    "MoEFCC": {
        "url": "https://moef.gov.in",
        "desc": "Ministry of Environment, Forest & Climate Change — national wildlife policies",
    },
    "India Biodiversity Portal": {
        "url": "https://indiabiodiversity.org",
        "desc": "Citizen science biodiversity observations across India",
    },
    "MoRTH Road Safety": {
        "url": "https://morth.nic.in",
        "desc": "Ministry of Road Transport — road accident statistics and highway data",
    },
    "WII — Wildlife Institute": {
        "url": "https://wii.gov.in",
        "desc": "Wildlife Institute of India — research publications, wildlife crossing studies",
    },
}

SPECIES_RISK_MAP = {
    'tiger': 0.9, 'elephant': 0.95, 'leopard': 0.85,
    'deer': 0.60, 'boar': 0.55, 'wolf': 0.70,
    'nilgai': 0.50, 'sambar': 0.65, 'gaur': 0.80,
    'sloth bear': 0.75, 'wild dog': 0.65, 'bison': 0.80,
}
ROAD_RISK_MAP = {
    'forest_road': 0.90, 'rural': 0.65, 'state_highway': 0.55,
    'highway': 0.45, 'national_highway': 0.40
}

WMO_VISIBILITY = {
    0: 950, 1: 900, 2: 850, 3: 750, 45: 200, 48: 100,
    51: 700, 53: 600, 55: 500, 61: 650, 63: 550, 65: 400,
    71: 400, 73: 300, 75: 200, 80: 600, 81: 500, 82: 350,
    95: 300, 96: 250, 99: 150,
}

WILDLIFE_KEYWORDS = [
    'elephant', 'tiger', 'leopard', 'deer', 'sambar', 'gaur', 'bison',
    'wild boar', 'bear', 'sloth bear', 'nilgai', 'wolf', 'wild dog',
    'wildlife', 'animal crossing', 'road kill', 'roadkill', 'animal hit',
    'wildlife corridor', 'highway crossing', 'animal accident',
]


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    la1, lo1, la2, lo2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = la2 - la1, lo2 - lo1
    a = sin(dlat/2)**2 + cos(la1)*cos(la2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


class ExtractionResult:
    def __init__(self, source_name, api_url, description):
        self.source_name = source_name
        self.api_url = api_url
        self.description = description
        self.status = "pending"
        self.timestamp = None
        self.response_ms = 0
        self.data = {}
        self.raw_preview = ""
        self.error_msg = ""
        self.features_extracted = []
        self.source_type = "api"  # api / news / government / computation

    def to_dict(self):
        return {
            "source": self.source_name, "api_url": self.api_url,
            "description": self.description, "status": self.status,
            "timestamp": self.timestamp, "response_ms": self.response_ms,
            "features": self.features_extracted, "error": self.error_msg,
            "raw_preview": self.raw_preview[:500] if self.raw_preview else "",
            "source_type": self.source_type,
        }


class RealtimeDataExtractor:
    def __init__(self):
        self.extraction_log: list[ExtractionResult] = []
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "WildGuardAI/3.0 (wildlife-risk-research; educational)"
        })

    # ══════════════════════════════════════════════════════════════════════════
    # 1. OPEN-METEO — Weather
    # ══════════════════════════════════════════════════════════════════════════
    def fetch_weather(self, lat, lon):
        api = (f"https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}"
               f"&current=temperature_2m,relative_humidity_2m,"
               f"precipitation,weather_code,wind_speed_10m,cloud_cover"
               f"&timezone=auto")
        result = ExtractionResult("Open-Meteo Weather API", api,
            "Real-time weather: temperature, humidity, precipitation, visibility")
        result.source_type = "api"
        try:
            t0 = time.time()
            resp = self._session.get(api, timeout=10)
            result.response_ms = round((time.time() - t0) * 1000)
            resp.raise_for_status()
            cur = resp.json().get("current", {})
            wc = cur.get("weather_code", 0)
            result.data = {
                "temperature_c": cur.get("temperature_2m", 28.0),
                "humidity_pct": cur.get("relative_humidity_2m", 65.0),
                "rainfall_mm": cur.get("precipitation", 0.0),
                "visibility_m": WMO_VISIBILITY.get(wc, 700),
                "weather_code": wc,
            }
            result.features_extracted = list(result.data.keys())
            result.status = "success"
            result.raw_preview = str(cur)[:400]
        except Exception as e:
            result.status = "fallback"
            result.error_msg = str(e)
            result.data = {"temperature_c": 28.0, "humidity_pct": 65.0,
                           "rainfall_mm": 5.0, "visibility_m": 700, "weather_code": 0}
            result.features_extracted = [f"{k} (fallback)" for k in result.data]
        result.timestamp = datetime.now().isoformat()
        self.extraction_log.append(result)
        return result.data

    # ══════════════════════════════════════════════════════════════════════════
    # 2. OVERPASS — Road & Water
    # ══════════════════════════════════════════════════════════════════════════
    def fetch_road_info(self, lat, lon):
        query = f"""[out:json][timeout:15];(
          way(around:1500,{lat},{lon})[highway];
          node(around:1500,{lat},{lon})[natural=water];
          way(around:1500,{lat},{lon})[waterway];
        );out body 30;"""
        result = ExtractionResult("OpenStreetMap Overpass API",
            "https://overpass-api.de/api/interpreter",
            "Road type, width, lighting, water body proximity from OSM")
        result.source_type = "api"
        osm_map = {'motorway':'national_highway','trunk':'national_highway',
                    'primary':'state_highway','secondary':'state_highway',
                    'tertiary':'rural','unclassified':'rural',
                    'residential':'rural','track':'forest_road','path':'forest_road'}
        try:
            t0 = time.time()
            resp = self._session.post("https://overpass-api.de/api/interpreter",
                                       data={"data": query.strip()}, timeout=15)
            result.response_ms = round((time.time() - t0) * 1000)
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            roads = [e for e in elements if e.get("type")=="way" and "highway" in e.get("tags",{})]
            waters = [e for e in elements if "water" in str(e.get("tags",{})) or "waterway" in str(e.get("tags",{}))]
            road_type, road_width, street_lighting = "rural", 7, 0
            if roads:
                tags = roads[0].get("tags", {})
                road_type = osm_map.get(tags.get("highway",""), "rural")
                street_lighting = 1 if tags.get("lit","no") in ("yes","24/7") else 0
                w = tags.get("width","")
                if w:
                    try: road_width = min(int(float(str(w).replace("m","").strip())), 14)
                    except: pass
            dist_water = 5.0
            for w in waters:
                if w.get("type") == "node":
                    dist_water = min(dist_water, _haversine(lat, lon, w["lat"], w["lon"]))
            result.data = {"road_type": road_type, "road_width_m": road_width,
                          "street_lighting": street_lighting, "curvature_deg_km": 12.0,
                          "dist_water_km": round(max(0.1, dist_water), 3)}
            result.features_extracted = list(result.data.keys())
            result.status = "success"
            result.raw_preview = str(elements[:2])[:400]
        except Exception as e:
            result.status = "fallback"
            result.error_msg = str(e)
            result.data = {"road_type":"rural","road_width_m":7,"street_lighting":0,
                          "curvature_deg_km":12.0,"dist_water_km":3.0}
            result.features_extracted = [f"{k} (fallback)" for k in result.data]
        result.timestamp = datetime.now().isoformat()
        self.extraction_log.append(result)
        return result.data

    # ══════════════════════════════════════════════════════════════════════════
    # 3. GBIF — Wildlife
    # ══════════════════════════════════════════════════════════════════════════
    def fetch_wildlife_data(self, lat, lon):
        api = (f"https://api.gbif.org/v1/occurrence/search"
               f"?decimalLatitude={lat-0.5},{lat+0.5}"
               f"&decimalLongitude={lon-0.5},{lon+0.5}"
               f"&classKey=359&limit=300&hasCoordinate=true")
        result = ExtractionResult("GBIF Biodiversity API", api,
            "Wildlife mammal occurrences within ~50km from GBIF.org")
        result.source_type = "api"
        try:
            t0 = time.time()
            resp = self._session.get(api, timeout=15)
            result.response_ms = round((time.time() - t0) * 1000)
            resp.raise_for_status()
            records = resp.json().get("results", [])
            species_counts = {}
            for rec in records:
                sp = (rec.get("species","") or rec.get("genus","") or "").lower()
                for our_sp in SPECIES_RISK_MAP:
                    if our_sp in sp:
                        species_counts[our_sp] = species_counts.get(our_sp, 0) + 1
            dominant = max(species_counts, key=species_counts.get) if species_counts else "deer"
            result.data = {"species": dominant, "species_risk": SPECIES_RISK_MAP.get(dominant, 0.6),
                          "total_sightings": len(records), "species_counts": species_counts}
            result.features_extracted = ["species","species_risk","total_sightings"]
            result.status = "success"
            result.raw_preview = str([{k:r.get(k) for k in ["species","decimalLatitude","eventDate"]} for r in records[:5]])[:400]
        except Exception as e:
            result.status = "fallback"
            result.error_msg = str(e)
            result.data = {"species":"deer","species_risk":0.60,"total_sightings":0,"species_counts":{}}
            result.features_extracted = [f"{k} (fallback)" for k in ["species","species_risk"]]
        result.timestamp = datetime.now().isoformat()
        self.extraction_log.append(result)
        return result.data

    # ══════════════════════════════════════════════════════════════════════════
    # 4. TEMPORAL — System Clock
    # ══════════════════════════════════════════════════════════════════════════
    def compute_temporal_features(self):
        result = ExtractionResult("System Clock (Temporal)", "datetime.now()",
            "Time-derived: hour, day, season, night/dawn/dusk/rush flags")
        result.source_type = "computation"
        now = datetime.now()
        h, m = now.hour, now.month
        season = ("summer" if m in [3,4,5] else "monsoon" if m in [6,7,8,9]
                  else "post_monsoon" if m in [10,11] else "winter")
        result.data = {
            "hour": h, "day_of_week": now.weekday(), "season": season,
            "night_flag": int(h < 6 or h >= 20),
            "dawn_dusk": int((5<=h<=7) or (17<=h<=19)),
            "rush_hour": int((6<=h<=9) or (17<=h<=20)),
            "breeding_season": int(season in ["monsoon","post_monsoon"]),
            "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        }
        result.features_extracted = ["hour","day_of_week","season","night_flag","dawn_dusk","rush_hour","breeding_season"]
        result.status = "success"
        result.timestamp = now.isoformat()
        result.raw_preview = f"Current: {now.isoformat()}"
        self.extraction_log.append(result)
        return result.data

    # ══════════════════════════════════════════════════════════════════════════
    # 5. GIS — Spatial Computation
    # ══════════════════════════════════════════════════════════════════════════
    def compute_spatial_features(self, lat, lon, season="monsoon"):
        result = ExtractionResult("GIS Computation Engine",
            "Haversine distance + NDVI estimation",
            "Distance to nearest protected area/corridor, NDVI estimate, night light index")
        result.source_type = "computation"
        t0 = time.time()
        pa_dists = sorted([(round(_haversine(lat,lon,p["lat"],p["lon"]),2), p["name"], p.get("state","")) for p in PROTECTED_AREAS])
        cor_dists = sorted([(round(_haversine(lat,lon,c["lat"],c["lon"]),2), c["name"]) for c in WILDLIFE_CORRIDORS])
        ndvi_base = max(0.15, 0.85 - pa_dists[0][0] * 0.02)
        season_mod = {"monsoon":0.12,"post_monsoon":0.05,"winter":-0.05,"summer":-0.10}
        ndvi = float(np.clip(ndvi_base + season_mod.get(season, 0), 0.05, 0.95))
        night_light = float(np.clip(15 + pa_dists[0][0]*3 + cor_dists[0][0]*2, 2, 200))
        result.data = {
            "protected_dist_km": pa_dists[0][0], "nearest_pa": pa_dists[0][1],
            "nearest_pa_state": pa_dists[0][2],
            "corridor_dist_km": cor_dists[0][0], "nearest_corridor": cor_dists[0][1],
            "ndvi": round(ndvi, 4), "night_light": round(night_light, 1),
            "top_3_pa": pa_dists[:3], "top_3_corridors": cor_dists[:3],
        }
        result.features_extracted = ["protected_dist_km","corridor_dist_km","ndvi","night_light"]
        result.response_ms = round((time.time()-t0)*1000)
        result.status = "success"
        result.timestamp = datetime.now().isoformat()
        result.raw_preview = f"Nearest PA: {pa_dists[0][1]} ({pa_dists[0][2]}, {pa_dists[0][0]}km) | Corridor: {cor_dists[0][1]} ({cor_dists[0][0]}km)"
        self.extraction_log.append(result)
        return result.data

    # ══════════════════════════════════════════════════════════════════════════
    # 6. NEWS WEBSITES — RSS Feeds (The Hindu, NDTV, Down to Earth, Google News)
    # ══════════════════════════════════════════════════════════════════════════
    def fetch_wildlife_news(self, lat, lon):
        result = ExtractionResult(
            "News Websites (RSS Aggregator)",
            "The Hindu | NDTV | Down to Earth | Google News RSS",
            "Scrapes wildlife-vehicle collision news from major Indian news sites via RSS. "
            "Searches for recent reports of animal-road incidents in South India."
        )
        result.source_type = "news"
        all_articles = []
        feed_results = {}
        t0 = time.time()

        # Find nearest state for region-specific filtering
        pa_dists = sorted([(_haversine(lat,lon,p["lat"],p["lon"]),p.get("state","")) for p in PROTECTED_AREAS])
        nearest_state = pa_dists[0][1] if pa_dists else "Karnataka"

        for feed_name, feed_url in NEWS_RSS_FEEDS.items():
            try:
                resp = self._session.get(feed_url, timeout=8)
                resp.raise_for_status()
                root = ET.fromstring(resp.content)
                items = root.findall('.//item')
                articles_from_feed = []
                for item in items[:30]:
                    title = (item.findtext('title') or '').strip()
                    desc = (item.findtext('description') or '').strip()
                    link = (item.findtext('link') or '').strip()
                    pub_date = (item.findtext('pubDate') or '').strip()
                    combined = f"{title} {desc}".lower()
                    # Check for wildlife keywords
                    matched_kw = [kw for kw in WILDLIFE_KEYWORDS if kw in combined]
                    if matched_kw:
                        # Check for species mentions
                        species_found = [sp for sp in SPECIES_RISK_MAP if sp in combined]
                        # Check for South India state mentions
                        south_states = ['karnataka','kerala','tamil nadu','andhra','telangana',
                                       'bandipur','wayanad','mudumalai','periyar','nagarhole',
                                       'nilgiri','western ghats','coorg','kodagu']
                        state_match = [s for s in south_states if s in combined]
                        articles_from_feed.append({
                            "title": title[:120],
                            "source": feed_name,
                            "url": link,
                            "date": pub_date[:25] if pub_date else "",
                            "keywords": matched_kw[:5],
                            "species_mentioned": species_found,
                            "south_india_match": bool(state_match),
                            "region_tags": state_match[:3],
                        })
                all_articles.extend(articles_from_feed)
                feed_results[feed_name] = f"{len(articles_from_feed)} wildlife articles from {len(items)} total"
            except Exception as e:
                feed_results[feed_name] = f"error: {str(e)[:60]}"

        result.response_ms = round((time.time()-t0)*1000)

        # Aggregate intelligence from news
        recent_incident_count = len(all_articles)
        south_india_incidents = sum(1 for a in all_articles if a.get("south_india_match"))
        species_in_news = {}
        for a in all_articles:
            for sp in a.get("species_mentioned", []):
                species_in_news[sp] = species_in_news.get(sp, 0) + 1

        result.data = {
            "total_wildlife_articles": recent_incident_count,
            "south_india_articles": south_india_incidents,
            "species_in_news": species_in_news,
            "top_articles": all_articles[:8],
            "feed_status": feed_results,
            "news_risk_modifier": min(0.15, recent_incident_count * 0.01),
        }
        result.features_extracted = [
            "total_wildlife_articles", "south_india_articles",
            "species_in_news", "news_risk_modifier"
        ]
        result.status = "success" if all_articles else "fallback"
        if not all_articles:
            result.error_msg = "No wildlife articles found in RSS feeds"
        result.timestamp = datetime.now().isoformat()
        result.raw_preview = str(feed_results)[:400] + "\n" + str(all_articles[:2])[:300]
        self.extraction_log.append(result)
        return result.data

    # ══════════════════════════════════════════════════════════════════════════
    # 7. GOVT OF INDIA — Forest Dept & Wildlife Portals
    # ══════════════════════════════════════════════════════════════════════════
    def fetch_govt_wildlife_data(self, lat, lon):
        result = ExtractionResult(
            "Indian Government Wildlife Portals",
            "NTCA | Karnataka/Kerala/TN Forest Depts | MoEFCC | WII | MoRTH | India Biodiversity Portal",
            "Fetches wildlife protection data, tiger reserve status, road safety statistics, "
            "and biodiversity records from Indian government and institutional portals."
        )
        result.source_type = "government"
        t0 = time.time()
        govt_data = {}
        source_status = {}

        # --- NTCA (National Tiger Conservation Authority) ---
        try:
            resp = self._session.get("https://ntca.gov.in", timeout=8)
            page_text = resp.text.lower()
            tiger_mentions = len(re.findall(r'tiger', page_text))
            corridor_mentions = len(re.findall(r'corridor', page_text))
            reserve_mentions = len(re.findall(r'reserve|sanctuary|national park', page_text))
            govt_data["ntca"] = {
                "accessible": True,
                "tiger_reserve_references": tiger_mentions,
                "corridor_references": corridor_mentions,
                "conservation_intensity": min(1.0, (tiger_mentions + corridor_mentions) / 100),
            }
            source_status["NTCA"] = f"✅ Accessible ({tiger_mentions} tiger refs, {corridor_mentions} corridor refs)"
        except Exception as e:
            govt_data["ntca"] = {"accessible": False, "conservation_intensity": 0.7}
            source_status["NTCA"] = f"⚠️ {str(e)[:50]}"

        # --- State Forest Departments (based on nearest state) ---
        pa_nearest = sorted([(_haversine(lat,lon,p["lat"],p["lon"]),p["name"],p.get("state","")) for p in PROTECTED_AREAS])
        nearest_state = pa_nearest[0][2] if pa_nearest else "Karnataka"

        state_urls = {
            "Karnataka": ("https://aranya.gov.in", "Karnataka Forest Dept"),
            "Kerala": ("https://forest.kerala.gov.in", "Kerala Forest Dept"),
            "Tamil Nadu": ("https://forests.tn.gov.in", "TN Forest Dept"),
            "Madhya Pradesh": ("https://mpforest.gov.in", "MP Forest Dept"),
            "Maharashtra": ("https://mahaforest.gov.in", "Maharashtra Forest Dept"),
        }
        state_info = state_urls.get(nearest_state, ("https://aranya.gov.in", "State Forest Dept"))
        try:
            resp = self._session.get(state_info[0], timeout=8)
            page_text = resp.text.lower()
            wildlife_refs = len(re.findall(r'wildlife|elephant|tiger|leopard|corridor', page_text))
            alert_refs = len(re.findall(r'alert|warning|advisory|caution', page_text))
            govt_data["state_forest"] = {
                "state": nearest_state, "accessible": True,
                "url": state_info[0], "dept_name": state_info[1],
                "wildlife_references": wildlife_refs,
                "alert_references": alert_refs,
                "active_alerts": alert_refs > 2,
            }
            source_status[state_info[1]] = f"✅ {wildlife_refs} wildlife refs, {alert_refs} alerts"
        except Exception as e:
            govt_data["state_forest"] = {"state": nearest_state, "accessible": False,
                                          "url": state_info[0], "dept_name": state_info[1]}
            source_status[state_info[1]] = f"⚠️ {str(e)[:50]}"

        # --- MoEFCC (Ministry of Environment) ---
        try:
            resp = self._session.get("https://moef.gov.in", timeout=8)
            govt_data["moefcc"] = {"accessible": True}
            source_status["MoEFCC"] = "✅ Accessible"
        except Exception as e:
            govt_data["moefcc"] = {"accessible": False}
            source_status["MoEFCC"] = f"⚠️ {str(e)[:50]}"

        # --- India Biodiversity Portal (attempt API-like access) ---
        try:
            ibp_url = f"https://indiabiodiversity.org/api/observation?lat={lat}&lng={lon}&radius=50"
            resp = self._session.get(ibp_url, timeout=8)
            if resp.status_code == 200:
                try:
                    ibp_data = resp.json()
                    govt_data["ibp"] = {"accessible": True, "observations": len(ibp_data) if isinstance(ibp_data, list) else 0}
                except:
                    govt_data["ibp"] = {"accessible": True, "observations": 0}
                source_status["India Biodiversity Portal"] = "✅ Accessible"
            else:
                govt_data["ibp"] = {"accessible": False}
                source_status["India Biodiversity Portal"] = f"⚠️ Status {resp.status_code}"
        except Exception as e:
            govt_data["ibp"] = {"accessible": False}
            source_status["India Biodiversity Portal"] = f"⚠️ {str(e)[:50]}"

        # --- MoRTH (Road Transport — accident data proxy) ---
        try:
            resp = self._session.get("https://morth.nic.in", timeout=8)
            page_text = resp.text.lower()
            accident_refs = len(re.findall(r'accident|road safety|fatality', page_text))
            govt_data["morth"] = {"accessible": True, "accident_references": accident_refs}
            source_status["MoRTH"] = f"✅ {accident_refs} safety refs"
        except Exception as e:
            govt_data["morth"] = {"accessible": False}
            source_status["MoRTH"] = f"⚠️ {str(e)[:50]}"

        # --- WII (Wildlife Institute of India) ---
        try:
            resp = self._session.get("https://wii.gov.in", timeout=8)
            page_text = resp.text.lower()
            research_refs = len(re.findall(r'research|study|publication|wildlife', page_text))
            govt_data["wii"] = {"accessible": True, "research_references": research_refs}
            source_status["WII"] = f"✅ {research_refs} research refs"
        except Exception as e:
            govt_data["wii"] = {"accessible": False}
            source_status["WII"] = f"⚠️ {str(e)[:50]}"

        result.response_ms = round((time.time()-t0)*1000)

        success_count = sum(1 for v in source_status.values() if v.startswith("✅"))
        conservation_intensity = govt_data.get("ntca",{}).get("conservation_intensity", 0.5)

        result.data = {
            "govt_sources_checked": len(source_status),
            "govt_sources_accessible": success_count,
            "source_status": source_status,
            "nearest_state": nearest_state,
            "nearest_state_dept": state_info[1],
            "nearest_state_url": state_info[0],
            "conservation_intensity": conservation_intensity,
            "active_alerts": govt_data.get("state_forest",{}).get("active_alerts", False),
            "detailed_data": govt_data,
        }
        result.features_extracted = [
            f"{success_count}/{len(source_status)} govt sources accessible",
            f"Nearest state: {nearest_state}",
            "conservation_intensity", "active_alerts"
        ]
        result.status = "success" if success_count >= 2 else "fallback"
        result.timestamp = datetime.now().isoformat()
        result.raw_preview = str(source_status)[:500]
        self.extraction_log.append(result)
        return result.data

    # ══════════════════════════════════════════════════════════════════════════
    # MASTER: Extract everything
    # ══════════════════════════════════════════════════════════════════════════
    def extract_all(self, lat, lon, speed_limit=60, actual_speed=65, past_accidents=3):
        self.extraction_log = []

        weather  = self.fetch_weather(lat, lon)
        road     = self.fetch_road_info(lat, lon)
        wildlife = self.fetch_wildlife_data(lat, lon)
        temporal = self.compute_temporal_features()
        spatial  = self.compute_spatial_features(lat, lon, temporal["season"])
        news     = self.fetch_wildlife_news(lat, lon)
        govt     = self.fetch_govt_wildlife_data(lat, lon)

        # Derived features
        speed_ratio = actual_speed / max(speed_limit, 1)
        road_risk = ROAD_RISK_MAP.get(road["road_type"], 0.55)
        driver_risk = speed_ratio * (1 + 0.6*temporal["night_flag"]) * road_risk
        movement_score = (0.30*spatial["ndvi"] + 0.25*min(1/(road["dist_water_km"]+0.1),1) +
                          0.20*temporal["dawn_dusk"] + 0.15*temporal["breeding_season"] +
                          0.10*(1-spatial["night_light"]/255))
        rolling_7day = past_accidents * np.random.uniform(0.8, 1.3)
        kde_density = past_accidents / (spatial["corridor_dist_km"]+0.5) * 0.4

        row = {
            "ndvi": spatial["ndvi"], "dist_water_km": road["dist_water_km"],
            "speed_limit": speed_limit, "actual_speed": actual_speed,
            "hour": temporal["hour"], "road_type": road["road_type"],
            "rainfall_mm": weather["rainfall_mm"], "visibility_m": int(weather["visibility_m"]),
            "past_accidents": past_accidents, "day_of_week": temporal["day_of_week"],
            "season": temporal["season"], "rush_hour": temporal["rush_hour"],
            "movement_score": round(float(movement_score),4),
            "kde_density": round(float(kde_density),4),
            "driver_risk": round(float(driver_risk),4),
            "corridor_dist_km": spatial["corridor_dist_km"],
            "breeding_season": temporal["breeding_season"],
            "species": wildlife["species"], "species_risk": wildlife["species_risk"],
            "night_light": spatial["night_light"],
            "rolling_7day": round(float(rolling_7day),2),
            "curvature_deg_km": road["curvature_deg_km"],
            "street_lighting": road["street_lighting"],
            "road_width_m": road["road_width_m"],
            "protected_dist_km": spatial["protected_dist_km"],
            "temperature_c": weather["temperature_c"],
            "humidity_pct": weather["humidity_pct"],
            "night_flag": temporal["night_flag"],
            "dawn_dusk": temporal["dawn_dusk"],
            "speed_ratio": round(speed_ratio, 3),
        }
        return pd.DataFrame([row])

    def get_extraction_log(self):
        return [r.to_dict() for r in self.extraction_log]

    def get_extraction_summary(self):
        total = len(self.extraction_log)
        return {
            "total_sources": total,
            "success_count": sum(1 for r in self.extraction_log if r.status=="success"),
            "fallback_count": sum(1 for r in self.extraction_log if r.status=="fallback"),
            "total_time_ms": sum(r.response_ms for r in self.extraction_log),
        }


if __name__ == "__main__":
    ext = RealtimeDataExtractor()
    df = ext.extract_all(11.66, 76.63)  # Bandipur, South India
    print("\n═══ Feature Vector ═══")
    print(df.T.to_string())
    print("\n═══ Extraction Log ═══")
    for l in ext.get_extraction_log():
        print(f"  [{l['status']:^8}] ({l['source_type']:^11}) {l['source']:<40} {l['response_ms']:>5}ms")
