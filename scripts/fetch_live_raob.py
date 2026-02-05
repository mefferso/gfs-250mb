#!/usr/bin/env python3
"""
scripts/fetch_live_raob.py

Fetches 00Z/12Z RAOB 250mb wind data directly from the University of Wyoming.
- Loads station list from stations/*.csv
- Determines correct Wyoming region based on Lat/Lon.
- Scrapes weather.uwyo.edu in parallel.

Outputs: data/raob/obs_latest.csv
"""

import csv
import datetime
import re
import sys
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
WORKER_COUNT = 10
TIMEOUT_SEC = 20


def _norm_header(h: str) -> str:
    """
    Normalize header names so:
      'Location Name' == 'LOCATION NAME' == 'location_name'
    """
    if h is None:
        return ""
    h = h.replace("\ufeff", "")          # BOM safety
    h = h.strip()
    h = h.replace("_", " ")
    h = re.sub(r"\s+", " ", h)           # collapse whitespace
    return h.upper()


def build_field_map(fieldnames):
    """
    Returns dict: NORMALIZED_HEADER -> original header
    """
    if not fieldnames:
        return {}
    fm = {}
    for fn in fieldnames:
        key = _norm_header(fn)
        if key and key not in fm:
            fm[key] = fn
    return fm


def load_stations():
    """
    Loads all stations from CSVs in the stations/ folder.
    Returns list of dicts: [{'id': '72250', 'lat': 25.9, 'lon': -97.4, 'name': '...'}, ...]
    """
    stations = []

    station_dir = Path("stations")
    if not station_dir.exists():
        station_dir = Path("../stations")

    if not station_dir.exists():
        print("ERROR: Could not find 'stations/' directory.")
        sys.exit(1)

    csv_files = list(station_dir.glob("*.csv"))
    print(f"Loading stations from {len(csv_files)} CSV files...")

    unique_ids = set()

    # Keys we’ll accept for each field
    id_candidates = [
        "STAT", "ID", "STATION", "WMO",
        "STATION ID", "STATION_ID",
        "ICAO", "ICAO ID", "ICAO_ID",
        "RAOB ID", "RAOB_ID",
        "IDENTIFIER",                 # <-- your column
    ]
    lat_candidates = ["LAT", "LATITUDE", "Y", "LAT DEG", "LAT_DEG", "LAT DEGREES", "LAT_DEGREES"]
    lon_candidates = ["LON", "LONGITUDE", "X", "LON DEG", "LON_DEG", "LON DEGREES", "LON_DEGREES"]
    name_candidates = [
        "NAME", "STATION NAME", "CITY", "DESCRIPTION",
        "LOCATION NAME",              # <-- your column
        "LOCATION_NAME",
    ]

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

    for csv_file in csv_files:
        try:
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                field_map = build_field_map(reader.fieldnames)

                print(f"  -> {csv_file.name} headers(norm): {list(field_map.keys())}")

                loaded_from_this_file = 0
                total_rows = 0
                missing_id = 0
                missing_lat = 0
                missing_lon = 0
                bad_float = 0

                for row in reader:
                    total_rows += 1

                    sid = get_val(row, field_map, id_candidates)
                    lat = get_val(row, field_map, lat_candidates)
                    lon = get_val(row, field_map, lon_candidates)
                    name = get_val(row, field_map, name_candidates) or sid

                    if not sid:
                        missing_id += 1
                        continue
                    if not lat:
                        missing_lat += 1
                        continue
                    if not lon:
                        missing_lon += 1
                        continue

                    clean_id = sid.strip().upper()
                    if clean_id in unique_ids:
                        continue

                    try:
                        lat_f = float(lat)
                        lon_f = float(lon)
                    except ValueError:
                        bad_float += 1
                        continue

                    stations.append({"id": clean_id, "lat": lat_f, "lon": lon_f, "name": name})
                    unique_ids.add(clean_id)
                    loaded_from_this_file += 1

                print(f"    Rows={total_rows} | loaded={loaded_from_this_file} | missing_id={missing_id} | missing_lat={missing_lat} | missing_lon={missing_lon} | bad_float={bad_float}")

        except Exception as e:
            print(f"Skipping {csv_file.name}: {e}")

    print(f"Loaded {len(stations)} unique stations in total.")
    return stations


def get_wyoming_region(lat, lon):
    if lat < -60:
        return "ant"
    if lat > 70:
        return "arctic"

    # North America
    if lat > 5 and -170 <= lon <= -50:
        return "naconf"

    # South America
    if lat <= 5 and -95 <= lon <= -25:
        return "samer"

    # Europe
    if lat > 30 and -25 <= lon <= 45:
        return "europe"

    # Africa
    if lat <= 35 and -20 <= lon <= 55:
        return "africa"

    # SE Asia / Australia / India
    if -50 <= lat <= 60 and 60 <= lon <= 180:
        return "seasia"

    # Middle East
    if 10 <= lat <= 60 and 30 <= lon <= 65:
        return "mideast"

    return "pac"


def get_target_cycle():
    now = datetime.datetime.utcnow()
    candidates = []

    if now.hour >= 13:
        candidates.append(now.replace(hour=12, minute=0, second=0, microsecond=0))
    if now.hour >= 1:
        candidates.append(now.replace(hour=0, minute=0, second=0, microsecond=0))

    candidates.append((now - datetime.timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0))
    candidates.append((now - datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0))

    return candidates


def fetch_station(station, target_dt):
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


def main():
    stations = load_stations()
    if not stations:
        print("ERROR: No stations loaded from CSVs. This is a header/format problem.")
        sys.exit(1)

    cycles = get_target_cycle()

    final_obs = []
    chosen_cycle = None

    for cycle in cycles:
        print(f"\n--- Attempting Cycle: {cycle} ---")
        obs_list = []

        with ThreadPoolExecutor(max_workers=WORKER_COUNT) as ex:
            futures = [ex.submit(fetch_station, s, cycle) for s in stations]

            done = 0
            for fut in as_completed(futures):
                done += 1
                if done % 50 == 0:
                    print(f"Progress: {done}/{len(stations)} checked...")
                res = fut.result()
                if res:
                    obs_list.append(res)

        print(f"Cycle {cycle}: Retrieved {len(obs_list)} observations.")

        if len(obs_list) > 20:
            final_obs = obs_list
            chosen_cycle = cycle
            break
        else:
            print("Insufficient data. Trying earlier cycle...")

    if not final_obs:
        print("ERROR: Failed to retrieve data for any cycle.")
        sys.exit(1)

    out_dir = Path("data/raob")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "obs_latest.csv"

    print(f"\nUsing cycle {chosen_cycle} — writing {len(final_obs)} rows to {out_file}...")
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["station", "name", "lat", "lon", "obs", "valid_utc"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in final_obs:
            w.writerow(row)

    print("Success.")


if __name__ == "__main__":
    main()
