#!/usr/bin/env python3
"""
scripts/fetch_live_raob.py

Fetches the latest available 00Z or 12Z RAOB data (250mb level) from IEM.
scans stations/*.csv to build a master lat/lon lookup to fix missing API coordinates.
Outputs: data/raob/obs_latest.csv
"""

import datetime
import sys
import json
import csv
import requests
from pathlib import Path

def load_station_metadata():
    """
    Loads ALL CSVs in the stations/ folder to build a master lat/lon map.
    This fixes "Null Island" issues by filling in missing API coordinates.
    """
    station_map = {}
    
    # Look for the stations directory relative to where the script is running
    # Github Actions runs from root, so "stations/" is correct.
    station_dir = Path("stations")
    
    if not station_dir.exists():
        # Fallback if running from inside scripts/ folder
        station_dir = Path("../stations")

    if not station_dir.exists():
        print("Warning: 'stations/' directory not found. Lat/Lon fallback disabled.")
        return {}

    # Find ALL .csv files (Europe.csv, Asia.csv, RAOB_250mb.csv, etc.)
    csv_files = list(station_dir.glob("*.csv"))
    
    if not csv_files:
        print("Warning: No CSV files found in stations/. Lat/Lon fallback disabled.")
        return {}

    print(f"Loading metadata from {len(csv_files)} station files...")

    for csv_file in csv_files:
        try:
            # utf-8-sig handles the BOM character if saved via Excel
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                
                # Normalize headers to upper case to handle "lat" vs "LAT" confusion
                # We do this by mapping the normalized field names
                field_map = {k.upper(): k for k in reader.fieldnames} if reader.fieldnames else {}
                
                # Helper to find a column ignoring case
                def get_val(row, *candidates):
                    for c in candidates:
                        key = field_map.get(c.upper())
                        if key and row.get(key):
                            return row[key]
                    return None

                count = 0
                for row in reader:
                    # Check common column names for Station ID
                    sid = get_val(row, 'STAT', 'ID', 'STATION', 'WMO')
                    
                    # Check common column names for Latitude
                    lat = get_val(row, 'LAT', 'LATITUDE', 'Y')
                    
                    # Check common column names for Longitude
                    lon = get_val(row, 'LON', 'LONGITUDE', 'X')
                    
                    if sid and lat and lon:
                        try:
                            # Store in map: KEY=STATION_ID, VAL=(LAT, LON)
                            station_map[sid.strip().upper()] = (float(lat), float(lon))
                            count += 1
                        except ValueError:
                            continue # Skip non-numeric lat/lon
                            
                print(f"  - Loaded {count} stations from {csv_file.name}")
                
        except Exception as e:
            print(f"  ! Error reading {csv_file.name}: {e}")
            
    print(f"Total unique stations loaded: {len(station_map)}")
    return station_map

def get_target_cycle():
    """
    Determines if we should look for today's 12Z, today's 00Z, or yesterday's 12Z.
    """
    now = datetime.datetime.utcnow()
    cycles = []
    
    # Candidate 1: Today 12Z (if we are past 13:30Z)
    if now.hour >= 13:
        cycles.append(now.replace(hour=12, minute=0, second=0, microsecond=0))
    # Candidate 2: Today 00Z (if we are past 01:30Z)
    if now.hour >= 1:
        cycles.append(now.replace(hour=0, minute=0, second=0, microsecond=0))
    # Candidate 3: Yesterday 12Z
    yesterday = now - datetime.timedelta(days=1)
    cycles.append(yesterday.replace(hour=12, minute=0, second=0, microsecond=0))
    # Candidate 4: Yesterday 00Z
    cycles.append(yesterday.replace(hour=0, minute=0, second=0, microsecond=0))

    return cycles

def fetch_iem_data(dt, station_map):
    """
    Fetches JSON from IEM for a specific timestamp.
    """
    ts_str = dt.strftime("%Y%m%d%H00") 
    url = f"https://mesonet.agron.iastate.edu/json/raob.py?ts={ts_str}&pressure=250"
    
    print(f"Attempting to fetch RAOBs for {ts_str} from IEM...")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Request failed: {e}")
        return None, 0

    if 'profiles' not in data:
        return None, 0
        
    rows = []
    for prof in data['profiles']:
        # Normalize station ID
        station = prof.get('station', '').strip().upper()
        items = prof.get('profile', [])
        
        # 1. Try to get Lat/Lon from API
        lat = prof.get('lat')
        lon = prof.get('lon')

        # 2. If missing or invalid (0,0), try local fallback from CSVs
        if not lat or not lon or (float(lat) == 0 and float(lon) == 0):
            if station in station_map:
                lat, lon = station_map[station]
            else:
                # If we absolutely can't find a location, skip it.
                # Plotting at 0,0 is worse than not plotting at all.
                continue

        for level in items:
            sknt = level.get('sknt')
            if sknt is not None:
                rows.append({
                    "station": station,
                    "valid_utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "lat": float(lat), 
                    "lon": float(lon),
                    "name": prof.get('name', station),
                    "obs": float(sknt)
                })
                break 
                
    return rows, len(rows)

def main():
    # 1. Load the master station map from your CSVs
    station_map = load_station_metadata()
    
    # 2. Determine which cycle (00Z/12Z) to fetch
    cycles = get_target_cycle()
    
    output_rows = []
    
    # 3. Try cycles until we find data
    for cycle in cycles:
        print(f"Checking cycle: {cycle}...")
        rows, count = fetch_iem_data(cycle, station_map)
        
        # We want a decent number of obs to consider it "complete"
        if count > 50: 
            print(f"Success! Found {count} observations for {cycle}.")
            output_rows = rows
            break
        else:
            print(f"Insufficient data ({count} obs). Trying previous cycle...")
            
    if not output_rows:
        print("ERROR: Could not fetch valid RAOB data for any recent cycle.")
        sys.exit(1)

    # 4. Save to CSV
    out_dir = Path("data/raob")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "obs_latest.csv"
    
    print(f"Writing {len(output_rows)} rows to {out_file}...")
    
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["station", "name", "lat", "lon", "obs", "valid_utc"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in output_rows:
            writer.writerow(r)
            
    print(f"OBS_IN={out_file}")

if __name__ == "__main__":
    main()
