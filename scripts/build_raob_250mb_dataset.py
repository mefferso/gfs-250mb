#!/usr/bin/env python3
"""
Build global 250mb RAOB vs model dataset for ONE valid time (VALID_UTC).
- Stations read from local CSVs in /stations
- GFS values sampled from the SAME GRIB used to make tiles (data/gfs.grib2)
- Apples-to-apples: RAOB must match valid time (±15 min), not "latest"

Outputs:
- data/raob/latest.json
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests
import xarray as xr

TARGET_MB = 250.0
KT_PER_MS = 1.94384
MAX_RAOB_OFFSET_MIN = 15

IEM_RAOB_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/raob.py"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

OPEN_METEO_MODELS = {
    "ECMWF": "ecmwf_ifs04",
    "CMC": "gem_global",
    "ICON": "icon_seamless",
}


# ---------------- station loading ----------------

def _lower_keys(d: dict) -> dict:
    return { (k or "").strip().lower(): v for k, v in d.items() }

def load_stations_from_local_csvs(stations_dir: str) -> List[dict]:
    files = [
        "RAOB_250mb.csv",
        "Europe.csv",
        "Asia.csv",
        "NorthAmerica_Africa_Austraila.csv",
    ]

    stations: List[dict] = []

    for fname in files:
        path = os.path.join(stations_dir, fname)
        if not os.path.exists(path):
            raise RuntimeError(f"Missing stations file: {path}")

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = _lower_keys(row)

                # flexible header reads
                name = r.get("location name") or r.get("location") or r.get("name") or r.get("station name") or ""
                ident = (r.get("identifier") or r.get("id") or r.get("station") or "").strip().upper()
                region = (r.get("region") or r.get("continent") or fname.replace(".csv", "")).strip()

                lat = r.get("latitude") or r.get("lat")
                lon = r.get("longitude") or r.get("lon") or r.get("long")

                if not ident or lat is None or lon is None:
                    continue

                stations.append({
                    "name": name or ident,
                    "region": region,
                    "id": ident,
                    "lat": float(lat),
                    "lon": float(lon),
                })

    # de-dupe by id + lat/lon
    uniq: Dict[Tuple[str, float, float], dict] = {}
    for s in stations:
        key = (s["id"], round(s["lat"], 4), round(s["lon"], 4))
        uniq[key] = s

    out = list(uniq.values())
    print(f"Loaded {len(out)} stations from local CSVs")
    return out


# ---------------- RAOB utils ----------------

def estimate_at_250(levels: List[Tuple[float, float]]) -> Optional[float]:
    """levels: list of (pressure_mb, speed_kts)"""
    # exact-ish
    for p, spd in levels:
        if abs(p - TARGET_MB) <= 0.5:
            return spd

    above = [x for x in levels if x[0] < TARGET_MB]  # lower pressure
    below = [x for x in levels if x[0] > TARGET_MB]  # higher pressure
    if above and below:
        p1, s1 = max(above, key=lambda x: x[0])  # closest above (highest p below target? careful)
        p2, s2 = min(below, key=lambda x: x[0])  # closest below
        # log-pressure interpolation
        w = (math.log(TARGET_MB) - math.log(p1)) / (math.log(p2) - math.log(p1))
        return s1 + (s2 - s1) * w

    return None


def fetch_iem_raob_250(station_id: str, target_dt: datetime) -> Optional[float]:
    """
    Query IEM RAOB CSV around target time and only accept a sounding within +/- 15 min.
    """
    sts = (target_dt - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%MZ")
    ets = (target_dt + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%MZ")
    url = f"{IEM_RAOB_URL}?station={station_id}&sts={sts}&ets={ets}"

    r = requests.get(url, timeout=60)
    if r.status_code != 200 or not r.text.strip():
        return None

    rows = list(csv.DictReader(r.text.splitlines()))
    groups: Dict[datetime, List[Tuple[float, float]]] = {}

    for row in rows:
        valid = (row.get("validUTC") or "").strip()
        if not valid:
            continue
        try:
            dt = datetime.strptime(valid, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        except Exception:
            continue

        try:
            p = float(row.get("pressure_mb") or "")
            spd = float(row.get("speed_kts") or row.get("sknt") or "")
        except Exception:
            continue

        groups.setdefault(dt, []).append((p, spd))

    # pick the dt that matches target within tolerance
    best_dt = None
    best_diff = None
    for dt in groups.keys():
        diff = abs((dt - target_dt).total_seconds()) / 60.0
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_dt = dt

    if best_dt is None or best_diff is None or best_diff > MAX_RAOB_OFFSET_MIN:
        return None

    return estimate_at_250(groups[best_dt])


# ---------------- model fetch ----------------

def fetch_open_meteo_models(lat: float, lon: float, target_dt: datetime) -> Dict[str, Optional[float]]:
    date = target_dt.strftime("%Y-%m-%d")
    tkey = target_dt.strftime("%Y-%m-%dT%H:00")

    url = (
        f"{OPEN_METEO_URL}?latitude={lat}&longitude={lon}"
        f"&start_date={date}&end_date={date}"
        f"&hourly=wind_speed_250hPa&wind_speed_unit=kn"
        f"&models=gfs_seamless,ecmwf_ifs04,gem_global,icon_seamless"
        f"&timezone=UTC"
    )

    out: Dict[str, Optional[float]] = {"ECMWF": None, "CMC": None, "ICON": None}

    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return out
        j = r.json()
        hourly = j.get("hourly") or {}
        times = hourly.get("time") or []
        if tkey not in times:
            return out
        idx = times.index(tkey)
        for label, suffix in OPEN_METEO_MODELS.items():
            arr = hourly.get(f"wind_speed_250hPa_{suffix}")
            if isinstance(arr, list) and idx < len(arr):
                out[label] = arr[idx]
    except Exception:
        return out

    return out


def sample_gfs_from_grib(grib_path: str, stations: List[dict]) -> Dict[str, Optional[float]]:
    """
    Sample 250mb wind speed (kt) from same GRIB used for tiles.
    Force isobaricInhPa to avoid multiple typeOfLevel conflict.
    """
    ds_u = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "shortName": "u"},
            "indexpath": ""
        },
    )
    ds_v = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "shortName": "v"},
            "indexpath": ""
        },
    )

    u = ds_u["u"].sel(isobaricInhPa=250)
    v = ds_v["v"].sel(isobaricInhPa=250)

    out: Dict[str, Optional[float]] = {}
    for s in stations:
        try:
            lon = s["lon"] if s["lon"] >= 0 else s["lon"] + 360.0
            uu = float(u.interp(latitude=s["lat"], longitude=lon))
            vv = float(v.interp(latitude=s["lat"], longitude=lon))
            out[s["id"]] = math.sqrt(uu * uu + vv * vv) * KT_PER_MS
        except Exception:
            out[s["id"]] = None

    return out


# ---------------- main ----------------

def main():
    valid = os.environ.get("VALID_UTC", "").strip()
    if not valid:
        raise RuntimeError("VALID_UTC env var is missing/empty")

    target_dt = datetime.strptime(valid, "%Y%m%d%H").replace(tzinfo=timezone.utc)

    stations = load_stations_from_local_csvs("stations")
    gfs = sample_gfs_from_grib("data/gfs.grib2", stations)

    rows: List[dict] = []
    for s in stations:
        obs = fetch_iem_raob_250(s["id"], target_dt)
        if obs is None:
            continue

        other = fetch_open_meteo_models(s["lat"], s["lon"], target_dt)

        def delta(model_val: Optional[float], obs_val: float) -> Optional[float]:
            if model_val is None:
                return None
            return float(model_val) - float(obs_val)

        gfs_val = gfs.get(s["id"])
        row = {
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
                "GFS":   {"speed": gfs_val,           "delta": delta(gfs_val, obs)},
                "ECMWF": {"speed": other["ECMWF"],    "delta": delta(other["ECMWF"], obs)},
                "CMC":   {"speed": other["CMC"],      "delta": delta(other["CMC"], obs)},
                "ICON":  {"speed": other["ICON"],     "delta": delta(other["ICON"], obs)},
            },
        }
        rows.append(row)

    out = {
        "meta": {
            "valid_utc": target_dt.strftime("%Y-%m-%dT%H:00Z"),
            "generated_utc": datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "station_count": len(stations),
            "raob_point_count": len(rows),
        },
        "rows": rows,
    }

    os.makedirs("data/raob", exist_ok=True)
    with open("data/raob/latest.json", "w", encoding="utf-8") as f:
        json.dump(out, f)

    print(f"Wrote {len(rows)} RAOB points to data/raob/latest.json")


if __name__ == "__main__":
    main()
