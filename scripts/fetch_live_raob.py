#!/usr/bin/env python3
"""
scripts/fetch_live_raob.py

Fetches 00Z/12Z RAOB 250mb wind data directly from the University of Wyoming.
- Loads station list from stations/*.csv
- Determines correct Wyoming region (naconf, europe, seasia, etc) based on Lat/Lon.
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
WORKER_COUNT = 10  # Number of parallel requests to Wyoming
TIMEOUT_SEC = 20   # Timeout per request


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

    # Helper to normalize header names
    def build_field_map(fieldnames):
        """
        Returns a dict mapping UPPER-CASED header -> original header string.
        """
        if not fieldnames:
            return {}
        return {fn.strip().upper(): fn for fn in fieldnames}

    for csv_file in csv_files:
        try:
            with open(csv_file, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                field_map = build_field_map(reader.fieldnames)

                print(f"  -> {csv_file.name} headers: {list(field_map.keys())}")

                # Keys we’ll accept for each field
                id_candidates = [
                    "STAT", "ID", "STATION", "WMO",
                    "STATION_ID", "ICAO", "ICAO_ID", "RAOB_ID"
                ]
                lat_candidates = [
                    "LAT", "LATITUDE", "Y", "LAT_DEG", "LAT_DEGREES"
                ]
                lon_candidates = [
                    "LON", "LONGITUDE", "X", "LON_DEG", "LON_DEGREES"
                ]
                name_candidates = [
                    "NAME", "STATION NAME", "CITY", "DESCRIPTION"
                ]

                def get_val(row, candidates):
                    for c in candidates:
                        if c in field_map:
                            raw_key = field_map[c]
                            val = row.get(raw_key)
                            if val not in (None, ""):
                                return val
                    return None

                loaded_from_this_file = 0

                for row in reader:
                    sid = get_val(row, id_candidates)
                    lat = get_val(row, lat_candidates)
                    lon = get_val(row, lon_candidates)
                    name = get_val(row, name_candidates) or sid

                    if not (sid and lat and lon):
                        # Row missing something critical; skip
                        continue

                    clean_id = sid.strip().upper()

                    if clean_id in unique_ids:
                        continue

                    try:
                        lat_f = float(lat)
                        lon_f = float(lon)
                    except ValueError:
                        # Bad numeric; skip this row
                        continue

                    stations.append(
                        {
                            "id": clean_id,
                            "lat": lat_f,
                            "lon": lon_f,
                            "name": name,
                        }
                    )
                    unique_ids.add(clean_id)
                    loaded_from_this_file += 1

                print(f"    Loaded {loaded_from_this_file} stations from {csv_file.name}")

        except Exception as e:
            print(f"Skipping {csv_file.name}: {e}")

    print(f"Loaded {len(stations)} unique stations in total.")
    return stations


def get_wyoming_region(lat, lon):
    """
    Maps Lat/Lon to Wyoming website region codes.
    """
    if lat < -60:
        return "ant"
    if lat > 70:
        # Overlap with europe/naconf, but arctic is safer for polar
        return "arctic"

    # North America
    if lat > 5 and -170 <= lon <= -50:
        return "naconf"

    # South America
    if lat <= 5 and -95 <= lon <= -25:
        return "samer"

    # Europe (Wide box)
    if lat > 30 and -25 <= lon <= 45:
        return "europe"

    # Africa
    if lat <= 35 and -20 <= lon <= 55:
        return "africa"

    # SE Asia / Australia / India
    # Wyoming 'seasia' map covers India through Australia
    if -50 <= lat <= 60 and 60 <= lon <= 180:
        return "seasia"

    # Middle East
    if 10 <= lat <= 60 and 30 <= lon <= 65:
        return "mideast"

    # Pacific (fallback)
    return "pac"


def get_target_cycle():
    """
    Returns a list of candidate datetimes to try (most recent first).
    """
    now = datetime.datetime.utcnow()
    candidates = []

    # Prioritize 12Z today if past 13:30Z
    if now.hour >= 13:
        candidates.append(now.replace(hour=12, minute=0, second=0, microsecond=0))
    # 00Z today
    if now.hour >= 1:
        candidates.append(now.replace(hour=0, minute=0, second=0, microsecond=0))
    # 12Z yesterday
    candidates.append(
        (now - datetime.timedelta(days=1)).replace(
            hour=12, minute=0, second=0, microsecond=0
        )
    )
    # 00Z yesterday
    candidates.append(
        (now - datetime.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    )

    return candidates


def fetch_station(station, target_dt):
    """
    Worker function to fetch a single station from Wyoming.
    """
    region = get_wyoming_region(station["lat"], station["lon"])

    # Params for Wyoming
    year = target_dt.strftime("%Y")
    month = target_dt.strftime("%m")
    # FROM/TO format is DDHH (e.g. 0500 or 0512)
    ddhh = target_dt.strftime("%d%H")
    stnm = station["id"]

    url = "http://weather.uwyo.edu/cgi-bin/sounding"
    params = {
        "region": region,
        "TYPE": "TEXT:LIST",
        "YEAR": year,
        "MONTH": month,
        "FROM": ddhh,
        "TO": ddhh,
        "STNM": stnm,
    }

    try:
        # Wyoming often slow, short timeout to fail fast
        r = requests.get(url, params=params, timeout=TIMEOUT_SEC)
        if r.status_code != 200:
            return None

        text = r.text

        # Parse HTML for 250mb
        # Look for lines starting with '  250.0'
        # Wyoming Columns: PRES HGHT TEMP DWPT RELH MIXR DRCT SKNT
        #   250.0 10240  -56.9  -68.9   34   0.03  265  105
        match = re.search(r"^\s*250\.0\s+.*", text, re.MULTILINE)
        if match:
            line = match.group(0)
            parts = line.split()
            # Parts: [250.0, HGHT, TEMP, DWPT, RELH, MIXR, DRCT, SKNT, ...]
            if len(parts) >= 8:
                try:
                    sknt = float(parts[7])
                    return {
                        "station": stnm,
                        "name": station["name"],
                        "lat": station["lat"],
                        "lon": station["lon"],
                        "obs": sknt,
                        "valid_utc": target_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                except ValueError:
                    return None
    except Exception:
        return None

    return None


def main():
    stations = load_stations()
    if not stations:
        print("ERROR: No stations loaded from CSVs. Check header names (ID/LAT/LON).")
        sys.exit(1)

    cycles = get_target_cycle()

    final_obs = []
    chosen_cycle = None

    # Try cycles until we get a "good" batch (>20 obs)
    for cycle in cycles:
        print(f"\n--- Attempting Cycle: {cycle} ---")
        obs_list = []

        with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
            future_to_station = {
                executor.submit(fetch_station, s, cycle): s for s in stations
            }

            completed = 0
            total = len(stations)

            for future in as_completed(future_to_station):
                completed += 1
                if completed % 50 == 0:
                    print(f"Progress: {completed}/{total} stations checked...")

                res = future.result()
                if res:
                    obs_list.append(res)

        print(f"Cycle {cycle}: Retrieved {len(obs_list)} observations.")

        if len(obs_list) > 20:  # Threshold for success
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in final_obs:
            writer.writerow(row)

    print(f"Success. OBS_IN={out_file}")


if __name__ == "__main__":
    main()
