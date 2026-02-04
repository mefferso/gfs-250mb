#!/usr/bin/env python3
"""
Build a global 250mb RAOB vs model dataset for ONE valid time (00Z or 12Z),
using local station CSVs committed to the repo.

STRICT RULES (apples-to-apples):
- ONE valid time only (00Z or 12Z)
- RAOB must match that valid time (±15 min max)
- GFS values come from the SAME GRIB used to make the tiles
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
import xarray as xr
import numpy as np

TARGET_MB = 250.0
KT_PER_MS = 1.94384
MAX_RAOB_OFFSET_MIN = 15

IEM_RAOB_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/raob.py"
UWYO_BASE = "http://weather.uwyo.edu/cgi-bin/sounding"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

OPEN_METEO_MODELS = {
    "ECMWF": "ecmwf_ifs04",
    "CMC": "gem_global",
    "ICON": "icon_seamless",
}


# ---------------- STATION LOADING ----------------

def load_stations_from_local_csvs(stations_dir: str):
    files = [
        "RAOB_250mb.csv",
        "Europe.csv",
        "Asia.csv",
        "NorthAmerica_Africa_Austraila.csv",
    ]

    stations = []

    for fname in files:
        path = os.path.join(stations_dir, fname)
        if not os.path.exists(path):
            raise RuntimeError(f"Missing stations file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stations.append({
                    "name": row.get("Location Name") or row.get("Name") or row.get("Station"),
                    "region": row.get("Region") or fname.replace(".csv", ""),
                    "id": row.get("Identifier") or row.get("ID"),
                    "lat": float(row["Latitude"]),
                    "lon": float(row["Longitude"]),
                })

    # de-duplicate
    uniq = {}
    for s in stations:
        key = (s["id"], round(s["lat"], 4), round(s["lon"], 4))
        uniq[key] = s

    print(f"Loaded {len(uniq)} stations")
    return list(uniq.values())


# ---------------- RAOB FETCHING ----------------

def estimate_at_250(levels):
    for p, spd in levels:
        if abs(p - TARGET_MB) < 0.5:
            return spd

    above = [x for x in levels if x[0] < TARGET_MB]
    below = [x for x in levels if x[0] > TARGET_MB]

    if above and below:
        p1, s1 = max(above)
        p2, s2 = min(below)
        w = (math.log(TARGET_MB) - math.log(p1)) / (math.log(p2) - math.log(p1))
        return s1 + (s2 - s1) * w

    return None


def fetch_iem_raob(station_id, target_dt):
    sts = (target_dt - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%MZ")
    ets = (target_dt + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%MZ")

    url = f"{IEM_RAOB_URL}?station={station_id}&sts={sts}&ets={ets}"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None

    rows = list(csv.DictReader(r.text.splitlines()))
    groups = {}

    for row in rows:
        valid = row.get("validUTC")
        if not valid:
            continue
        dt = datetime.strptime(valid, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        groups.setdefault(dt, []).append((float(row["pressure_mb"]), float(row["speed_kts"])))

    for dt, levels in groups.items():
        if abs((dt - target_dt).total_seconds()) / 60 <= MAX_RAOB_OFFSET_MIN:
            return estimate_at_250(levels)

    return None


# ---------------- MODEL FETCHING ----------------

def fetch_open_meteo(lat, lon, target_dt):
    date = target_dt.strftime("%Y-%m-%d")
    hour = target_dt.strftime("%H:00")

    url = (
        f"{OPEN_METEO_URL}?latitude={lat}&longitude={lon}"
        f"&start_date={date}&end_date={date}"
        f"&hourly=wind_speed_250hPa&wind_speed_unit=kn"
        f"&models=gfs_seamless,ecmwf_ifs04,gem_global,icon_seamless"
    )

    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return {}

    j = r.json()
    times = j["hourly"]["time"]
    idx = times.index(f"{date}T{hour}")

    out = {}
    for label, key in OPEN_METEO_MODELS.items():
        out[label] = j["hourly"].get(f"wind_speed_250hPa_{key}", [None])[idx]

    return out


def sample_gfs_from_grib(grib_path, stations):
    ds_u = xr.open_dataset(grib_path, engine="cfgrib",
                           backend_kwargs={"filter_by_keys": {"shortName": "u"}})
    ds_v = xr.open_dataset(grib_path, engine="cfgrib",
                           backend_kwargs={"filter_by_keys": {"shortName": "v"}})

    u = ds_u["u"].sel(isobaricInhPa=250)
    v = ds_v["v"].sel(isobaricInhPa=250)

    out = {}
    for s in stations:
        lon = s["lon"] if s["lon"] >= 0 else s["lon"] + 360
        uu = float(u.interp(latitude=s["lat"], longitude=lon))
        vv = float(v.interp(latitude=s["lat"], longitude=lon))
        out[s["id"]] = math.sqrt(uu**2 + vv**2) * KT_PER_MS

    return out


# ---------------- MAIN ----------------

def main():
    valid = os.environ["VALID_UTC"]
    target_dt = datetime.strptime(valid, "%Y%m%d%H").replace(tzinfo=timezone.utc)

    stations = load_stations_from_local_csvs("stations")
    gfs = sample_gfs_from_grib("data/gfs.grib2", stations)

    rows = []

    for s in stations:
        obs = fetch_iem_raob(s["id"], target_dt)
        if obs is None:
            continue

        models = fetch_open_meteo(s["lat"], s["lon"], target_dt)

        rows.append({
            "name": s["name"],
            "region": s["region"],
            "id": s["id"],
            "lat": s["lat"],
            "lon": s["lon"],
            "date": target_dt.strftime("%Y-%m-%d"),
            "time": target_dt.strftime("%H:00"),
            "cycle": target_dt.strftime("%HZ"),
            "obs": obs,
            "models": {
                "GFS": {"speed": gfs.get(s["id"]), "delta": gfs.get(s["id"]) - obs},
                "ECMWF": {"speed": models.get("ECMWF"), "delta": (models.get("ECMWF") - obs) if models.get("ECMWF") else None},
                "CMC": {"speed": models.get("CMC"), "delta": (models.get("CMC") - obs) if models.get("CMC") else None},
                "ICON": {"speed": models.get("ICON"), "delta": (models.get("ICON") - obs) if models.get("ICON") else None},
            }
        })

    os.makedirs("data/raob", exist_ok=True)
    with open("data/raob/latest.json", "w") as f:
        json.dump({"rows": rows}, f)

    print(f"Wrote {len(rows)} RAOB points")


if __name__ == "__main__":
    main()
