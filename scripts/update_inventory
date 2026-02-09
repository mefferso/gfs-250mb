import os
import json
import shutil
import datetime
from pathlib import Path

# --- Configuration ---
HISTORY_DIR = Path("data/history")
TILES_GFS_DIR = Path("tiles/250mb")      # GFS tiles storage
TILES_ECMWF_DIR = Path("tiles/ecmwf")    # ECMWF tiles storage
KEEP_DAYS = 5                            # Keep 5 days of history

def main():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Scan for valid cycle JSONs (Format: YYYYMMDDHH.json)
    # We look for files where the name (stem) is just digits (e.g. 2026020912)
    files = [f for f in HISTORY_DIR.glob("*.json") if f.stem.isdigit()]
    
    inventory = []
    
    # Calculate cutoff for deletion (older than 5 days)
    cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=KEEP_DAYS)
    
    for f in sorted(files, reverse=True):
        cycle_str = f.stem # e.g. "2026020912"
        
        try:
            # Parse the filename into a date object
            dt = datetime.datetime.strptime(cycle_str, "%Y%m%d%H")
        except ValueError:
            continue # Skip files that don't match the date format

        # 2. Cleanup Old Data
        if dt < cutoff_date:
            print(f"Removing old cycle: {cycle_str}")
            # Delete JSON
            f.unlink()
            # Delete GFS Tiles folder
            gfs_path = TILES_GFS_DIR / cycle_str
            if gfs_path.exists(): shutil.rmtree(gfs_path)
            # Delete ECMWF Tiles folder
            ec_path = TILES_ECMWF_DIR / cycle_str
            if ec_path.exists(): shutil.rmtree(ec_path)
            continue

        # 3. Add to inventory list
        inventory.append({
            "cycle": cycle_str,
            "label": dt.strftime("%Y-%m-%d %Hz"),
            "path": f"data/history/{cycle_str}.json",
            "gfs_tiles": f"tiles/250mb/{cycle_str}/{{z}}/{{x}}/{{y}}.png",
            "ecmwf_tiles": f"tiles/ecmwf/{cycle_str}/{{z}}/{{x}}/{{y}}.png"
        })

    # Write the inventory file for the website to read
    # This is the file your website was missing!
    with open("data/history/inventory.json", "w") as f:
        json.dump(inventory, f, indent=2)
    
    print(f"Inventory updated. {len(inventory)} cycles available.")

if __name__ == "__main__":
    main()
