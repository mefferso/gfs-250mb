#!/usr/bin/env python3
"""
scripts/fetch_live_raob.py

Fetches the latest available 00Z or 12Z RAOB data (250mb level) from IEM.
Outputs: data/raob/obs_latest.csv
"""

import datetime
import sys
import json
import csv
import requests
from pathlib import Path

def get_target_cycle():
    """
    Determines if we should look for today's 12Z, today's 00Z, or yesterday's 12Z.
    RAOBs usually arrive by ~14:00 UTC (for 12Z) and ~02:00 UTC (for 00Z).
    """
    now = datetime.datetime.utcnow()
    
    # Simple logic: If it's past 14Z, try fetching 12Z. 
    # If it's past 02Z, try fetching 00Z. 
    # Otherwise, fallback to yesterday's 12Z.
    
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

def fetch_iem_data(dt):
    """
    Fetches JSON from IEM for a specific timestamp.
    """
    ts_str = dt.strftime("%Y%m%d%H00") # e.g., 202602051200
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
        
    # Filter valid rows
    rows = []
    for prof in data['profiles']:
        # IEM JSON structure: 'station', 'profile': [{'p': 250, 'sknt': 85, ...}]
        # Sometimes the profile is a list of dicts.
        station = prof.get('station')
        items = prof.get('profile', [])
        
        # Find the 250mb level (or close to it if needed, but IEM filtering usually handles it)
        # The URL param &pressure=250 requests interpolation/nearest.
        
        for level in items:
            sknt = level.get('sknt')
            if sknt is not None:
                rows.append({
                    "station": station,
                    "valid_utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "lat": prof.get('lat', 0), # IEM sometimes puts lat/lon at root
                    "lon": prof.get('lon', 0),
                    "name": prof.get('name', station),
                    "obs": float(sknt)
                })
                break # only need one 250mb reading per station
                
    return rows, len(rows)

def main():
    cycles = get_target_cycle()
    
    output_rows = []
    found_cycle = None
    
    for cycle in cycles:
        print(f"Checking cycle: {cycle}...")
        rows, count = fetch_iem_data(cycle)
        if count > 50: # Threshold to ensure we didn't just get a partial empty update
            print(f"Success! Found {count} observations for {cycle}.")
            output_rows = rows
            found_cycle = cycle
            break
        else:
            print(f"Insufficient data ({count} obs). Trying previous cycle...")
            
    if not output_rows:
        print("ERROR: Could not fetch valid RAOB data for any recent cycle.")
        sys.exit(1)

    # Save to CSV
    out_dir = Path("data/raob")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "obs_latest.csv"
    
    print(f"Writing {len(output_rows)} rows to {out_file}...")
    
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["station", "name", "lat", "lon", "obs", "valid_utc"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in output_rows:
            # IEM usually gives lat/lon in the station metadata, 
            # if missing in JSON we might need a station list, but usually it's there.
            # Handle edge case if lat/lon missing
            if not r.get('lat'): r['lat'] = 0.0
            if not r.get('lon'): r['lon'] = 0.0
            
            writer.writerow(r)
            
    print(f"OBS_IN={out_file}")

if __name__ == "__main__":
    main()
