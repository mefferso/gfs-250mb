#!/usr/bin/env python3
import json
import shutil
import datetime
from pathlib import Path

# --- Configuration ---
HISTORY_DIR = Path("data/history")
STAGING_DIR = Path("data/raob/staging")

# Tile roots by model (match publish_raob_latest.py tiles-map)
TILES_DIRS = {
    "GFS": Path("tiles/250mb"),
    "ECMWF": Path("tiles/ecmwf"),
    # Add later:
    # "CMC": Path("tiles/cmc"),
    # "ICON": Path("tiles/icon"),
}

KEEP_DAYS = 5  # keep 5 days of published cycles


def rm_tree(p: Path):
    if p.exists():
        shutil.rmtree(p)


def main():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    # Published combined cycles are data/history/YYYYMMDDHH.json
    files = [f for f in HISTORY_DIR.glob("*.json") if f.stem.isdigit()]
    inventory = []

    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=KEEP_DAYS)

    for f in sorted(files, reverse=True):
        cycle = f.stem  # YYYYMMDDHH

        try:
            dt = datetime.datetime.strptime(cycle, "%Y%m%d%H")
        except ValueError:
            continue

        # Cleanup older than cutoff
        if dt < cutoff:
            print(f"Removing old published cycle: {cycle}")
            f.unlink()

            # remove tiles cycle folders for each model
            for model, root in TILES_DIRS.items():
                rm_tree(root / cycle)

            # remove staging cycle folder
            rm_tree(STAGING_DIR / cycle)
            continue

        # Verify tile folders exist for all configured models
        tiles_ok = True
        tiles_paths = {}
        for model, root in TILES_DIRS.items():
            p = root / cycle
            if not p.exists():
                tiles_ok = False
            tiles_paths[model] = f"{root.as_posix()}/{cycle}/{{z}}/{{x}}/{{y}}.png"

        if not tiles_ok:
            # If history exists but tiles are missing, skip inventory entry
            continue

        entry = {
            "cycle": cycle,
            "label": dt.strftime("%Y-%m-%d %Hz"),
            "path": f"data/history/{cycle}.json",
        }

        # Keep old keys for your site (gfs_tiles/ecmwf_tiles)
        if "GFS" in tiles_paths:
            entry["gfs_tiles"] = tiles_paths["GFS"]
        if "ECMWF" in tiles_paths:
            entry["ecmwf_tiles"] = tiles_paths["ECMWF"]

        inventory.append(entry)

    (HISTORY_DIR / "inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(f"Inventory updated. {len(inventory)} published cycles available.")


if __name__ == "__main__":
    main()
