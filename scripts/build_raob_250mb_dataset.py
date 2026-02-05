#!/usr/bin/env python3
"""
scripts/build_raob_250mb_dataset.py

- Reads RAOB obs (CSV/JSON).
- Loads EXISTING latest.json (if valid) to preserve other models.
- Samples 250mb SPEED GeoTIFFs for currently available models.
- Updates and writes data/raob/latest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MS_TO_KT = 1.9438444924406

_USE_RASTERIO = False
_USE_GDAL = False

try:
    import numpy as np
    import rasterio
    _USE_RASTERIO = True
except Exception:
    try:
        import numpy as np
        from osgeo import gdal
        _USE_GDAL = True
    except Exception:
        pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, str) and x.strip() == "":
        return None
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def normalize_station_id(x: Any) -> str:
    return str(x or "").strip().upper()


@dataclass
class ObsRow:
    station: str
    name: str
    lat: float
    lon: float
    obs_kt: float
    valid_utc: str = ""


def read_rows_from_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def read_rows_from_json(path: Path) -> Tuple[Any, List[Dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload, payload["rows"]
    if isinstance(payload, list):
        return payload, payload
    raise ValueError(f"JSON format not recognized: {path}")


def load_obs(path: Path) -> Tuple[str, List[ObsRow]]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows_raw: List[Dict[str, Any]] = []
    valid_guess = ""

    if path.suffix.lower() == ".csv":
        rows_raw = read_rows_from_csv(path)
        # Try to find valid_utc in first row
        if rows_raw and "valid_utc" in rows_raw[0]:
            valid_guess = rows_raw[0]["valid_utc"]
    else:
        payload, rows_raw = read_rows_from_json(path)
        valid_guess = payload.get("meta", {}).get("valid_utc", "")

    out: List[ObsRow] = []
    for r in rows_raw:
        station = normalize_station_id(r.get("station") or r.get("id"))
        if not station:
            continue

        name = str(r.get("name") or "").strip()
        lat = to_float_or_none(r.get("lat"))
        lon = to_float_or_none(r.get("lon"))
        obs = to_float_or_none(r.get("obs") or r.get("obs_kt"))
        vutc = str(r.get("valid_utc") or "").strip()

        if lat is None or lon is None or obs is None:
            continue

        out.append(ObsRow(station, name, lat, lon, obs, vutc))

    return valid_guess, out


@dataclass
class RasterInfo:
    path: Path
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    nodata: Optional[float]
    backend: str
    units: str


def open_raster(path: Path, units: str) -> RasterInfo:
    units = (units or "kt").lower()
    units_norm = "ms" if units in ("ms", "m/s", "mps") else "kt"

    if _USE_RASTERIO:
        with rasterio.open(path) as ds:
            b = ds.bounds
            return RasterInfo(
                path=path,
                lon_min=float(b.left),
                lon_max=float(b.right),
                lat_min=float(b.bottom),
                lat_max=float(b.top),
                nodata=ds.nodata,
                backend="rasterio",
                units=units_norm,
            )
    raise RuntimeError("No raster backend available.")


def _adjust_lon(lon: float, info: RasterInfo) -> float:
    if info.lon_min >= 0 and lon < 0:
        lon = lon % 360.0
    if info.lon_max <= 180 and lon > 180:
        lon = ((lon + 180) % 360) - 180
    return lon


def sample_points(info: RasterInfo, points: List[Tuple[float, float]]) -> List[Optional[float]]:
    vals: List[Optional[float]] = [None] * len(points)
    if info.backend == "rasterio":
        with rasterio.open(info.path) as ds:
            coords = []
            for lat, lon in points:
                lon2 = _adjust_lon(lon, info)
                coords.append((float(lon2), float(lat)))

            for i, arr in enumerate(ds.sample(coords)):
                try:
                    v = float(arr[0])
                    if info.nodata is not None and v == info.nodata:
                        vals[i] = None
                    elif not np.isfinite(v):
                        vals[i] = None
                    else:
                        if info.units == "ms":
                            v *= MS_TO_KT
                        vals[i] = float(v)
                except Exception:
                    vals[i] = None
    return vals


def merge_json_data(
    obs_rows: List[ObsRow],
    valid_utc: str,
    model_rasters: Dict[str, Optional[RasterInfo]],
    out_path: Path,
) -> None:
    """
    Merging Strategy:
    1. Load existing JSON from out_path (if it exists).
    2. Check if existing JSON 'valid_utc' matches current 'valid_utc'.
    3. If match: Pre-fill model data from existing JSON to preserve models we aren't running right now.
    4. If no match: Start fresh (all models null).
    5. Overwrite data for models we DO have rasters for in this run.
    """
    
    # --- 1. Load Existing Data ---
    existing_data_map = {} # station_id -> { "GFS": {speed, delta}, "ECMWF": ... }
    
    if out_path.exists():
        try:
            old_payload = json.loads(out_path.read_text(encoding="utf-8"))
            old_meta = old_payload.get("meta", {})
            old_valid = old_meta.get("valid_utc", "")
            
            # --- 2. Check Timestamp ---
            if old_valid == valid_utc:
                print(f"[INFO] Existing JSON timestamp ({old_valid}) matches. Merging data.")
                for r in old_payload.get("rows", []):
                    sid = normalize_station_id(r.get("id") or r.get("station"))
                    existing_data_map[sid] = r.get("models", {})
            else:
                print(f"[INFO] Existing JSON timestamp ({old_valid}) != Current ({valid_utc}). Starting fresh.")
        except Exception as e:
            print(f"[WARN] Failed to read existing JSON: {e}")

    # --- 3. Sample New Rasters ---
    coords = [(r.lat, r.lon) for r in obs_rows]
    sampled_new: Dict[str, List[Optional[float]]] = {}

    for model, rinfo in model_rasters.items():
        if rinfo is not None:
            print(f"[INFO] Sampling {model} from {rinfo.path}...")
            sampled_new[model] = sample_points(rinfo, coords)
        else:
            sampled_new[model] = None

    # --- 4. Build Output Rows ---
    rows_out: List[Dict[str, Any]] = []
    
    for i, r in enumerate(obs_rows):
        # Start with existing models for this station (if any)
        models_obj = existing_data_map.get(r.station, {})
        
        # Ensure keys exist if fresh
        for m in ["GFS", "ECMWF", "CMC", "ICON"]:
            if m not in models_obj:
                models_obj[m] = {"speed": None, "delta": None}

        # Overwrite with new samples
        for model in ["GFS", "ECMWF", "CMC", "ICON"]:
            # If we have a raster for this model in THIS run, update the value.
            # If we don't have a raster (rinfo is None), KEEP the old value.
            if sampled_new.get(model) is not None:
                spd = sampled_new[model][i]
                delta = (spd - r.obs_kt) if (spd is not None and r.obs_kt is not None) else None
                models_obj[model] = {"speed": spd, "delta": delta}

        rows_out.append({
            "name": r.name,
            "id": r.station,
            "lat": r.lat,
            "lon": r.lon,
            "valid_utc": valid_utc,
            "obs": r.obs_kt,
            "models": models_obj,
        })

    # --- 5. Write ---
    # Update metadata: calculate 'models_present' based on what is not null in the first row
    models_present = {k: False for k in ["GFS", "ECMWF", "CMC", "ICON"]}
    if rows_out:
        for m in models_present.keys():
            # If any row has a value, we consider it present
            if any(row["models"][m]["speed"] is not None for row in rows_out):
                models_present[m] = True

    payload = {
        "meta": {
            "valid_utc": valid_utc,
            "generated_utc": utc_now_iso(),
            "station_count": len(rows_out),
            "models_present": models_present,
        },
        "rows": rows_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=None), encoding="utf-8") # Minified for size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", help="Input obs CSV")
    ap.add_argument("--out", dest="out_path", default="data/raob/latest.json")
    
    ap.add_argument("--gfs-tif", default="")
    ap.add_argument("--ecmwf-tif", default="")
    ap.add_argument("--cmc-tif", default="")
    ap.add_argument("--icon-tif", default="")
    
    ap.add_argument("--gfs-units", default="kt")
    ap.add_argument("--ecmwf-units", default="kt")
    ap.add_argument("--cmc-units", default="kt")
    ap.add_argument("--icon-units", default="kt")

    args = ap.parse_args()

    if not _USE_RASTERIO:
        print("ERROR: need rasterio", file=sys.stderr)
        return 2

    # 1. Load Obs
    if not args.in_path or not Path(args.in_path).exists():
        print(f"ERROR: Input file {args.in_path} not found.", file=sys.stderr)
        return 2
        
    valid_guess, obs_rows = load_obs(Path(args.in_path))
    if not obs_rows:
        print("ERROR: 0 obs parsed", file=sys.stderr)
        return 2

    # 2. Determine Valid Time
    valid_utc = valid_guess
    if not valid_utc:
        # Fallback to obs row time
        valid_utc = obs_rows[0].valid_utc
    
    print(f"Dataset Valid Time: {valid_utc}")

    # 3. Open Rasters (Safely)
    def safe_open(path_str, units):
        if not path_str or not Path(path_str).exists():
            return None
        try:
            return open_raster(Path(path_str), units)
        except Exception as e:
            print(f"[WARN] Failed open {path_str}: {e}")
            return None

    model_rasters = {
        "GFS": safe_open(args.gfs_tif, args.gfs_units),
        "ECMWF": safe_open(args.ecmwf_tif, args.ecmwf_units),
        "CMC": safe_open(args.cmc_tif, args.cmc_units),
        "ICON": safe_open(args.icon_tif, args.icon_units),
    }

    # 4. Merge and Build
    merge_json_data(obs_rows, valid_utc, model_rasters, Path(args.out_path))
    print(f"Success. Wrote {args.out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
