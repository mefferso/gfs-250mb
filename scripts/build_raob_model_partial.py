#!/usr/bin/env python3
"""
scripts/build_raob_model_partial.py

Build a *staged* partial RAOB dataset for ONE model.

Input:
  - Obs CSV from scripts/fetch_live_raob.py (data/raob/obs_latest.csv)
  - A model 250mb speed GeoTIFF (kt or m/s)

Output (example):
  data/raob/staging/2026020912/GFS.json

This does NOT touch data/raob/latest.json.
Publishing happens later (atomic) by scripts/publish_raob_latest.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MS_TO_KT = 1.9438444924406

try:
    import numpy as np
    import rasterio
except Exception as e:
    print(f"ERROR: rasterio/numpy required: {e}", file=sys.stderr)
    raise


ALL_MODELS = ["GFS", "ECMWF", "CMC", "ICON"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso_utc(s: str) -> Optional[datetime]:
    s = (s or "").strip()
    if not s:
        return None
    # allow "2026-02-09T12:00:00Z"
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def cycle_from_valid_utc(valid_utc: str) -> str:
    dt = parse_iso_utc(valid_utc)
    if not dt:
        return ""
    return dt.astimezone(timezone.utc).strftime("%Y%m%d%H")


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
    valid_utc: str


def read_obs_csv(path: Path) -> Tuple[str, List[ObsRow]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return "", []

    # valid_utc should be consistent across rows (00Z or 12Z),
    # but we’ll just trust the first row.
    valid_utc = (rows[0].get("valid_utc") or "").strip()

    out: List[ObsRow] = []
    for r in rows:
        station = normalize_station_id(r.get("station") or r.get("id"))
        if not station:
            continue

        name = str(r.get("name") or "").strip()
        lat = to_float_or_none(r.get("lat"))
        lon = to_float_or_none(r.get("lon"))
        obs = to_float_or_none(r.get("obs") or r.get("obs_kt"))
        vutc = (r.get("valid_utc") or valid_utc or "").strip()

        if lat is None or lon is None or obs is None:
            continue

        out.append(ObsRow(station=station, name=name, lat=lat, lon=lon, obs_kt=obs, valid_utc=vutc))

    return valid_utc, out


@dataclass
class RasterInfo:
    path: Path
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    nodata: Optional[float]
    units: str  # "kt" or "ms"


def open_raster(path: Path, units: str) -> RasterInfo:
    units = (units or "kt").lower()
    units_norm = "ms" if units in ("ms", "m/s", "mps") else "kt"

    with rasterio.open(path) as ds:
        b = ds.bounds
        return RasterInfo(
            path=path,
            lon_min=float(b.left),
            lon_max=float(b.right),
            lat_min=float(b.bottom),
            lat_max=float(b.top),
            nodata=ds.nodata,
            units=units_norm,
        )


def _adjust_lon(lon: float, info: RasterInfo) -> float:
    # handle datasets that are 0..360 vs -180..180
    if info.lon_min >= 0 and lon < 0:
        lon = lon % 360.0
    if info.lon_max <= 180 and lon > 180:
        lon = ((lon + 180) % 360) - 180
    return lon


def sample_points(info: RasterInfo, points_latlon: List[Tuple[float, float]]) -> List[Optional[float]]:
    vals: List[Optional[float]] = [None] * len(points_latlon)

    with rasterio.open(info.path) as ds:
        coords = []
        for lat, lon in points_latlon:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="data/raob/obs_latest.csv", help="Obs CSV from fetch_live_raob.py")
    ap.add_argument("--model", required=True, help="Model name (GFS, ECMWF, CMC, ICON)")
    ap.add_argument("--tif", required=True, help="Model 250mb speed GeoTIFF")
    ap.add_argument("--units", default="kt", help="kt or ms (m/s)")
    ap.add_argument("--out", default="", help="Output staged partial JSON. If blank, auto under staging/<cycle>/<MODEL>.json")

    args = ap.parse_args()

    model = args.model.strip().upper()
    if model not in ALL_MODELS:
        print(f"ERROR: model must be one of {ALL_MODELS}", file=sys.stderr)
        return 2

    obs_path = Path(args.obs)
    if not obs_path.exists():
        print(f"ERROR: obs CSV not found: {obs_path}", file=sys.stderr)
        return 2

    tif_path = Path(args.tif)
    if not tif_path.exists():
        print(f"ERROR: tif not found: {tif_path}", file=sys.stderr)
        return 2

    valid_utc, obs_rows = read_obs_csv(obs_path)
    if not valid_utc or not obs_rows:
        print("ERROR: no obs rows parsed / missing valid_utc", file=sys.stderr)
        return 2

    cycle = cycle_from_valid_utc(valid_utc)
    if not cycle:
        print(f"ERROR: could not compute cycle from valid_utc={valid_utc}", file=sys.stderr)
        return 2

    out_path = Path(args.out) if args.out else Path(f"data/raob/staging/{cycle}/{model}.json")

    rinfo = open_raster(tif_path, args.units)
    coords = [(r.lat, r.lon) for r in obs_rows]
    sampled = sample_points(rinfo, coords)

    rows_out: List[Dict[str, Any]] = []
    for i, r in enumerate(obs_rows):
        spd = sampled[i]
        delta = (spd - r.obs_kt) if (spd is not None) else None

        rows_out.append({
            "name": r.name,
            "id": r.station,
            "lat": r.lat,
            "lon": r.lon,
            "valid_utc": valid_utc,
            "obs": r.obs_kt,
            "models": {
                model: {"speed": spd, "delta": delta}
            }
        })

    payload = {
        "meta": {
            "cycle": cycle,
            "valid_utc": valid_utc,
            "generated_utc": utc_now_iso(),
            "model": model,
            "station_count": len(rows_out),
        },
        "rows": rows_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    print(f"[OK] wrote staged partial: {out_path} (cycle={cycle}, n={len(rows_out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
