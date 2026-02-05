#!/usr/bin/env python3
"""
scripts/fetch_live_raob.py

Fetch live 250 mb wind (kt) observations.

- Stations listed in RAOB_250mb.csv (your US list) are fetched via IEMCow:
    https://mesonet.agron.iastate.edu/json/raob.py  (pressure=250)
- All other stations (from stations/*.csv) are fetched via UWyo:
    http://weather.uwyo.edu/cgi-bin/sounding

Output:
  data/raob/obs_latest.csv
"""

import csv
import datetime
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# --- CONFIG ---
WORKER_COUNT = 12
TIMEOUT_SEC = 20


# -----------------------------
# Header normalization helpers
# -----------------------------
def _norm_header(h: str) -> str:
    if h is None:
        return ""
    h = h.replace("\ufeff", "")  # BOM safety
    h = h.strip()
    h = h.replace("_", " ")
    h = re.sub(r"\s+", " ", h)  # collapse whitespace
    return h.upper()


def build_field_map(fieldnames):
    """dict: NORMALIZED_HEADER -> original header"""
    if not fieldnames:
        return {}
    fm = {}
    for fn in fieldnames:
        key = _norm_header(fn)
        if key and key not in fm:
            fm[key] = fn
    return fm


def get_val(row, field_map, candidates):
    for c in candidates:
        c_norm = _norm_header(c)
        if c_norm in field_map:
            raw_key = field_map[c_norm]
            val = row.get(raw_key)
            if val is not None:
                val = str(val).strip()
                if val != "":
                    return val
    return None


# -----------------------------
# Load RAOB_250mb.csv (IEM list)
# -----------------------------
def load_raob250mb_iem_list() -> dict:
    """
    Returns dict keyed by station id (Identifier) with station metadata.
    Each returned station has prefer_iem=True.
    """
    candidates = [
        Path("RAOB_250mb.csv"),
        Path("data/raob/RAOB_250mb.csv"),
        Path("stations/RAOB_250mb.csv"),
        Path("../RAOB_250mb.csv"),
        Path("../data/raob/RAOB_250mb.csv"),
        Path("../stations/RAOB_250mb.csv"),
    ]
    path = next((p for p in candidates if p.exists()), None)
    if not path:
        return {}

    stations = {}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        field_map = build_field_map(reader.fieldnames)

        id_candidates = ["IDENTIFIER", "ID", "STATION", "STAT", "ICAO", "RAOB ID", "RAOB_ID"]
        name_candidates = ["LOCATION NAME", "NAME", "STATION NAME", "CITY", "DESCRIPTION"]
        lat_candidates = ["LAT", "LATITUDE", "Y"]
        lon_candidates = ["LON", "LONGITUDE", "X"]
        source_candidates = ["RAOB SOURCE", "SOURCE"]

        for row in reader:
            sid = get_val(row, field_map, id_candidates)
            lat = get_val(row, field_map, lat_candidates)
            lon = get_val(row, field_map, lon_candidates)
            name = get_val(row, field_map, name_candidates) or sid
            src = (get_val(row, field_map, source_candidates) or "").strip().upper()

            if not (sid and lat and lon):
                continue

            # Only treat these as "IEM list" if the file says IEM (or if source is blank, still allow)
            # If you want *all* rows in RAOB_250mb.csv to be IEM, keep as-is.
            if src and "IEM" not in src:
                continue

            try:
                lat_f = float(lat)
                lon_f = float(lon)
            except ValueError:
                continue

            clean_id = sid.strip().upper()
            stations[clean_id] = {
                "id": clean_id,
                "lat": lat_f,
                "lon": lon_f,
                "name": name,
                "prefer_iem": True,
            }

    print(f"Loaded {len(stations)} IEM stations from {path}")
    return stations


# -----------------------------
# Load stations/*.csv (UWyo list)
# -----------------------------
def load_stations() -> list:
    """
    Loads all stations from:
      - RAOB_250mb.csv (prefer_iem=True)
      - stations/*.csv  (prefer_iem=False)

    Returns list of dicts:
      {'id': 'KJAX', 'lat': 30.49, 'lon': -81.7, 'name': 'Jacksonville', 'prefer_iem': True/False}
    """
    stations_by_id = {}

    # 1) RAOB_250mb.csv stations (IEM preferred)
    stations_by_id.update(load_raob250mb_iem_list())

    # 2) stations/*.csv stations (UWyo default)
    station_dir = Path("stations")
    if not station_dir.exists():
        station_dir = Path("../stations")

    if not station_dir.exists():
        if stations_by_id:
            # We can still run with just RAOB_250mb.csv
            return list(stations_by_id.values())
        print("ERROR: Could not find 'stations/' directory and no RAOB_250mb.csv found.")
        sys.exit(1)

    csv_files = [p for p in station_dir.glob("*.csv") if p.name.lower() != "raob_250mb.csv"]
    print(f"Loading stations from {len(csv_files)} CSV files in {station_dir}...")

    # Keys we’ll accept for each field
    id_candidates = [
        "STAT", "ID", "STATION", "WMO",
        "STATION ID", "STATION_ID",
        "ICAO", "ICAO ID", "ICAO_ID",
        "RAOB ID", "RAOB_ID",
        "IDENTIFIER",
    ]
    lat_candidates = ["LAT", "LATITUDE", "Y", "LAT DEG", "LAT_DEG", "LAT DEGREES", "LAT_DEGREES"]
    lon_candidates = ["LON", "LONGITUDE", "X", "LON DEG", "LON_DEG", "LON DEGREES", "LON_DEGREES"]
    name_candidates = ["NAME", "STATION NAME", "CITY", "DESCRIPTION", "LOCATION NAME", "LOCATION_NAME"]

    for csv_file in csv_files:
        try:
            with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                field_map = build_field_map(reader.fieldnames)

                loaded_from_this_file = 0
                for row in reader:
                    sid = get_val(row, field_map, id_candidates)
                    lat = get_val(row, field_map, lat_candidates)
                    lon = get_val(row, field_map, lon_candidates)
                    name = get_val(row, field_map, name_candidates) or sid

                    if not (sid and lat and lon):
                        continue

                    clean_id = sid.strip().upper()
                    if clean_id in stations_by_id:
                        # Keep IEM preference if already set from RAOB_250mb.csv
                        continue

                    try:
                        lat_f = float(lat)
                        lon_f = float(lon)
                    except ValueError:
                        continue

                    stations_by_id[clean_id] = {
                        "id": clean_id,
                        "lat": lat_f,
                        "lon": lon_f,
                        "name": name,
                        "prefer_iem": False,
                    }
                    loaded_from_this_file += 1

                print(f"  -> {csv_file.name}: loaded {loaded_from_this_file}")

        except Exception as e:
            print(f"Skipping {csv_file.name}: {e}")

    print(f"Loaded {len(stations_by_id)} unique stations total.")
    return list(stations_by_id.values())


# -----------------------------
# Time logic
# -----------------------------
def get_target_cycles():
    """
    Returns candidate datetimes (UTC) most-recent first.
    """
    now = datetime.datetime.utcnow()
    cycles = []

    # prefer 12Z when it's actually available
    if now.hour >= 13:
        cycles.append(now.replace(hour=12, minute=0, second=0, microsecond=0))
    # prefer 00Z when it's actually available
    if now.hour >= 1:
        cycles.append(now.replace(hour=0, minute=0, second=0, microsecond=0))

    # fallbacks
    yday = now - datetime.timedelta(days=1)
    cycles.append(yday.replace(hour=12, minute=0, second=0, microsecond=0))
    cycles.append(yday.replace(hour=0, minute=0, second=0, microsecond=0))

    # de-dupe while preserving order
    out = []
    seen = set()
    for c in cycles:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


# -----------------------------
# IEMCow fetch (RAOB_250mb.csv stations)
# -----------------------------
def fetch_station_iem(station, target_dt):
    """
    Fetch 250mb wind speed (kt) using IEM JSON RAOB service.
    """
    base_url = "https://mesonet.agron.iastate.edu/json/raob.py"
    ts = target_dt.strftime("%Y%m%d%H00")  # accepted by IEM

    def _try_station_id(sid):
        params = {"station": sid, "ts": ts, "pressure": 250}
        r = requests.get(base_url, params=params, timeout=TIMEOUT_SEC)
        if r.status_code != 200:
            return None
        j = r.json()
        profiles = j.get("profiles") or []
        if not profiles:
            return None
        prof = profiles[0]
        pts = prof.get("profile") or []
        if not pts:
            return None

        # Find best point (exact pres==250, else nearest to 250)
        best = None
        best_dp = 999999
        for pt in pts:
            pres = pt.get("pres")
            sknt = pt.get("sknt")
            if pres is None or sknt is None:
                continue
            try:
                dp = abs(float(pres) - 250.0)
            except Exception:
                continue
            if dp < best_dp:
                best_dp = dp
                best = pt

        if not best or best.get("sknt") is None:
            return None

        valid = (prof.get("valid") or "").strip()
        if valid:
            # IEM returns "YYYY-MM-DD T%H:%M:%SZ" (note space before T)
            valid = valid.replace(" ", "")
        else:
            valid = target_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        return {
            "station": station["id"],
            "name": station["name"],
            "lat": station["lat"],
            "lon": station["lon"],
            "obs": float(best["sknt"]),
            "valid_utc": valid,
        }

    sid = station["id"].strip().upper()

    # Try as-is first
    out = _try_station_id(sid)
    if out:
        return out

    # If it's Kxxx, also try xxx (IEM allows 3-letter for K***)
    if len(sid) == 4 and sid.startswith("K"):
        out = _try_station_id(sid[1:])
        if out:
            return out

    return None


# -----------------------------
# UWyo fetch (all other stations)
# -----------------------------
def get_wyoming_region(lat, lon):
    if lat < -60:
        return "ant"
    if lat > 70:
        return "arctic"

    if lat > 5 and -170 <= lon <= -50:
        return "naconf"
    if lat <= 5 and -95 <= lon <= -25:
        return "samer"
    if lat > 30 and -25 <= lon <= 45:
        return "europe"
    if lat <= 35 and -20 <= lon <= 55:
        return "africa"
    if -50 <= lat <= 60 and 60 <= lon <= 180:
        return "seasia"
    if 10 <= lat <= 60 and 30 <= lon <= 65:
        return "mideast"
    return "pac"


def fetch_station_uwyo(station, target_dt):
    """
    Fetch 250mb wind speed (kt) using UWyo TEXT:LIST.
    """
    region = get_wyoming_region(station["lat"], station["lon"])
    year = target_dt.strftime("%Y")
    month = target_dt.strftime("%m")
    ddhh = target_dt.strftime("%d%H")

    url = "http://weather.uwyo.edu/cgi-bin/sounding"
    params = {
        "region": region,
        "TYPE": "TEXT:LIST",
        "YEAR": year,
        "MONTH": month,
        "FROM": ddhh,
        "TO": ddhh,
        "STNM": station["id"],
    }

    try:
        r = requests.get(url, params=params, timeout=TIMEOUT_SEC)
        if r.status_code != 200:
            return None

        text = r.text
        match = re.search(r"^\s*250\.0\s+.*", text, re.MULTILINE)
        if not match:
            return None

        parts = match.group(0).split()
        if len(parts) < 8:
            return None

        sknt = float(parts[7])
        return {
            "station": station["id"],
            "name": station["name"],
            "lat": station["lat"],
            "lon": station["lon"],
            "obs": sknt,
            "valid_utc": target_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    except Exception:
        return None


def fetch_station(station, target_dt):
    if station.get("prefer_iem"):
        return fetch_station_iem(station, target_dt)
    return fetch_station_uwyo(station, target_dt)


# -----------------------------
# Main
# -----------------------------
def main():
    print('Fetching latest RAOBs...')
    stations = load_stations()
    if not stations:
        print("ERROR: No stations loaded.")
        sys.exit(1)

    cycles = get_target_cycles()
    final_obs = []
    chosen_cycle = None

    for cycle in cycles:
        print(f"\n--- Attempting Cycle: {cycle.strftime('%Y-%m-%d %H:00Z')} ---")
        obs_list = []

        with ThreadPoolExecutor(max_workers=WORKER_COUNT) as ex:
            futures = [ex.submit(fetch_station, s, cycle) for s in stations]

            done = 0
            for fut in as_completed(futures):
                done += 1
                if done % 100 == 0:
                    print(f"Progress: {done}/{len(stations)} checked...")
                res = fut.result()
                if res:
                    obs_list.append(res)

        print(f"Cycle {cycle.strftime('%Y-%m-%d %H:00Z')}: Retrieved {len(obs_list)} observations.")

        # success threshold
        if len(obs_list) > 20:
            final_obs = obs_list
            chosen_cycle = cycle
            break

        print("Insufficient data. Trying earlier cycle...")

    if not final_obs:
        print("ERROR: Failed to retrieve data for any cycle.")
        sys.exit(1)

    out_dir = Path("data/raob")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "obs_latest.csv"

    print(f"\nUsing cycle {chosen_cycle.strftime('%Y-%m-%d %H:00Z')} — writing {len(final_obs)} rows to {out_file}...")
    with out_file.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["station", "name", "lat", "lon", "obs", "valid_utc"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in final_obs:
            w.writerow(row)

    print("Success.")


if __name__ == "__main__":
    main()
