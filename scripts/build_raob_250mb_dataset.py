#!/usr/bin/env python3
"""
build_raob_250mb_dataset.py

Builds data/raob/latest.json for the web map.

Inputs:
  - A stations/obs file (CSV or JSON) that contains, per row:
      station (or id), name, lat, lon, obs  (obs is RAOB 250mb wind speed in knots)

  - Model SPEED GeoTIFFs (single-band float/real) for:
      GFS, ECMWF, CMC, ICON
    IMPORTANT: These must be the *raw speed* rasters, NOT colorized RGB.

Output JSON (default):
  data/raob/latest.json

Usage examples:
  python build_raob_250mb_dataset.py \
    --in data/raob/obs_latest.csv \
    --valid-utc 2026-02-04T00:00Z \
    --gfs-tif output_250mb_speed.tif \
    --ecmwf-tif ecmwf_250mb_speed.tif \
    --cmc-tif cmc_250mb_speed.tif \
    --icon-tif icon_250mb_speed.tif

If you don't pass model tif paths, it will try to auto-find likely files via glob.
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -------------------------
# Optional raster backends
# -------------------------
_USE_RASTERIO = False
try:
    import numpy as np  # type: ignore
    import rasterio  # type: ignore
    _USE_RASTERIO = True
except Exception:
    _USE_RASTERIO = False

_USE_GDAL = False
if not _USE_RASTERIO:
    try:
        from osgeo import gdal  # type: ignore
        _USE_GDAL = True
    except Exception:
        _USE_GDAL = False


MS_TO_KT = 1.9438444924406


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def to_float_or_none(x: Any) -> Optional[float]:
    # Critical: treat None/"" as missing (NOT 0)
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
    s = str(x or "").strip()
    return s.upper()


def guess_valid_utc_from_anything(payload: Any) -> str:
    # Try common spots if input is JSON from another step
    if isinstance(payload, dict):
        meta = payload.get("meta") or {}
        for k in ("valid_utc", "validUtc", "valid", "time"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # maybe top-level
        for k in ("valid_utc", "validUtc", "valid"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


@dataclass
class ObsRow:
    station: str
    name: str
    lat: float
    lon: float
    obs_kt: float
    valid_utc: str = ""


# -------------------------
# Input readers (CSV/JSON)
# -------------------------
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


def detect_and_load_obs(path: Path) -> Tuple[str, List[ObsRow]]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    valid_guess = ""
    rows_raw: List[Dict[str, Any]] = []

    if path.suffix.lower() == ".csv":
        rows_raw = read_rows_from_csv(path)
    elif path.suffix.lower() in (".json", ".geojson"):
        payload, rows_raw = read_rows_from_json(path)
        valid_guess = guess_valid_utc_from_anything(payload)
    else:
        raise ValueError(f"Unsupported input type: {path.suffix}")

    out: List[ObsRow] = []
    for r in rows_raw:
        station = normalize_station_id(
            r.get("station") or r.get("id") or r.get("identifier") or r.get("wmo") or r.get("icao")
        )
        name = str(r.get("name") or r.get("location") or r.get("locName") or r.get("location_name") or "").strip()

        lat = to_float_or_none(r.get("lat") or r.get("latitude"))
        lon = to_float_or_none(r.get("lon") or r.get("longitude"))
        obs = to_float_or_none(r.get("obs") or r.get("raob_speed_kt") or r.get("obs_kt") or r.get("raob"))

        # per-row valid if present
        vutc = str(r.get("valid_utc") or r.get("validUtc") or r.get("valid") or "").strip()

        if not station or lat is None or lon is None or obs is None:
            continue

        out.append(ObsRow(station=station, name=name, lat=lat, lon=lon, obs_kt=obs, valid_utc=vutc))

    # if the rows had no per-row valid, keep guessed meta valid
    return valid_guess, out


# -------------------------
# Raster sampling
# -------------------------
@dataclass
class RasterInfo:
    path: Path
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    nodata: Optional[float]
    backend: str  # "rasterio" or "gdal"
    units: str    # "kt" or "ms"


def _resolve_candidate_paths(explicit: Optional[str], patterns: List[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        return None

    # try patterns
    for pat in patterns:
        hits = list(Path(".").glob(pat))
        hits = [h for h in hits if h.is_file()]
        if hits:
            # pick newest
            hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return hits[0]
    return None


def open_raster(path: Path, units: str) -> RasterInfo:
    units = (units or "kt").lower()
    if units not in ("kt", "kn", "kts", "knots", "ms", "m/s", "mps"):
        units = "kt"
    units_norm = "ms" if units in ("ms", "m/s", "mps") else "kt"

    if _USE_RASTERIO:
        with rasterio.open(path) as ds:
            b = ds.bounds  # left, bottom, right, top
            nodata = ds.nodata
            return RasterInfo(
                path=path,
                lon_min=float(b.left),
                lon_max=float(b.right),
                lat_min=float(b.bottom),
                lat_max=float(b.top),
                nodata=None if nodata is None else float(nodata),
                backend="rasterio",
                units=units_norm,
            )

    if _USE_GDAL:
        ds = gdal.Open(str(path))
        if ds is None:
            raise RuntimeError(f"GDAL could not open: {path}")
        gt = ds.GetGeoTransform()
        # gt: originX, pixelW, rot1, originY, rot2, pixelH(negative)
        x0, px_w, _, y0, _, px_h = gt
        x1 = x0 + px_w * ds.RasterXSize
        y1 = y0 + px_h * ds.RasterYSize
        lon_min, lon_max = (min(x0, x1), max(x0, x1))
        lat_min, lat_max = (min(y0, y1), max(y0, y1))
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        return RasterInfo(
            path=path,
            lon_min=float(lon_min),
            lon_max=float(lon_max),
            lat_min=float(lat_min),
            lat_max=float(lat_max),
            nodata=None if nodata is None else float(nodata),
            backend="gdal",
            units=units_norm,
        )

    raise RuntimeError("No raster backend available. Install rasterio or GDAL.")


def _adjust_lon_for_raster(lon: float, info: RasterInfo) -> float:
    """
    Handles 0..360 vs -180..180 differences.
    """
    # If raster is clearly 0..360 and lon is negative, wrap.
    if info.lon_min >= 0 and lon < 0:
        lon = lon % 360.0
    # If raster is clearly -180..180 and lon is > 180, unwrap.
    if info.lon_max <= 180 and lon > 180:
        lon = ((lon + 180) % 360) - 180
    return lon


def sample_raster_points(info: RasterInfo, points: List[Tuple[float, float]]) -> List[Optional[float]]:
    """
    points: list of (lat, lon)
    returns list of sampled values (knots) or None
    """
    vals: List[Optional[float]] = [None] * len(points)

    if info.backend == "rasterio":
        # Open once, sample many
        with rasterio.open(info.path) as ds:
            coords = []
            for (lat, lon) in points:
                lon2 = _adjust_lon_for_raster(lon, info)
                coords.append((float(lon2), float(lat)))  # rasterio wants (x=lon, y=lat)

            for i, arr in enumerate(ds.sample(coords)):
                try:
                    v = float(arr[0])
                except Exception:
                    vals[i] = None
                    continue

                if info.nodata is not None and v == info.nodata:
                    vals[i] = None
                    continue
                if not np.isfinite(v):
                    vals[i] = None
                    continue

                if info.units == "ms":
                    v *= MS_TO_KT

                vals[i] = float(v)

        return vals

    if info.backend == "gdal":
        ds = gdal.Open(str(info.path))
        if ds is None:
            return vals
        band = ds.GetRasterBand(1)
        gt = ds.GetGeoTransform()

        # Inverse transform
        success, inv_gt = gdal.InvGeoTransform(gt)
        if not success:
            return vals

        for i, (lat, lon) in enumerate(points):
            lon2 = _adjust_lon_for_raster(lon, info)
            px, py = gdal.ApplyGeoTransform(inv_gt, float(lon2), float(lat))
            col, row = int(px), int(py)

            if col < 0 or row < 0 or col >= ds.RasterXSize or row >= ds.RasterYSize:
                vals[i] = None
                continue

            arr = band.ReadAsArray(col, row, 1, 1)
            if arr is None:
                vals[i] = None
                continue
            v = float(arr[0][0])

            if info.nodata is not None and v == info.nodata:
                vals[i] = None
                continue
            if not math.isfinite(v):
                vals[i] = None
                continue

            if info.units == "ms":
                v *= MS_TO_KT

            vals[i] = float(v)

        return vals

    return vals


# -------------------------
# Main builder
# -------------------------
def build_latest_json(
    obs_rows: List[ObsRow],
    valid_utc: str,
    model_rasters: Dict[str, Optional[RasterInfo]],
    out_path: Path,
) -> None:
    coords = [(r.lat, r.lon) for r in obs_rows]

    sampled: Dict[str, List[Optional[float]]] = {}
    for model, rinfo in model_rasters.items():
        if rinfo is None:
            sampled[model] = [None] * len(obs_rows)
            continue
        sampled[model] = sample_raster_points(rinfo, coords)

    rows_out: List[Dict[str, Any]] = []
    for i, r in enumerate(obs_rows):
        models_obj: Dict[str, Any] = {}
        for model in ("ECMWF", "CMC", "ICON", "GFS"):
            spd = sampled.get(model, [None] * len(obs_rows))[i]
            delta = (spd - r.obs_kt) if (spd is not None and r.obs_kt is not None) else None
            models_obj[model] = {"speed": spd, "delta": delta}

        rows_out.append(
            {
                "station": r.station,
                "name": r.name,
                "lat": r.lat,
                "lon": r.lon,
                "valid_utc": r.valid_utc or valid_utc,
                "obs": r.obs_kt,
                "models": models_obj,
            }
        )

    payload = {
        "meta": {
            "valid_utc": valid_utc,
            "generated_utc": utc_now_iso(),
            "count_obs": len(rows_out),
            "models_present": {k: (v is not None) for k, v in model_rasters.items()},
        },
        "rows": rows_out,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="", help="Input obs file (CSV or JSON) with station/name/lat/lon/obs(kt)")
    ap.add_argument("--out", dest="out_path", default="data/raob/latest.json", help="Output JSON path")
    ap.add_argument("--valid-utc", dest="valid_utc", default="", help="Valid time e.g. 2026-02-04T00:00Z")

    ap.add_argument("--gfs-tif", default="", help="GFS 250mb SPEED GeoTIFF (single-band)")
    ap.add_argument("--ecmwf-tif", default="", help="ECMWF 250mb SPEED GeoTIFF (single-band)")
    ap.add_argument("--cmc-tif", default="", help="CMC/GEM 250mb SPEED GeoTIFF (single-band)")
    ap.add_argument("--icon-tif", default="", help="ICON 250mb SPEED GeoTIFF (single-band)")

    ap.add_argument("--gfs-units", default="kt", help="kt (default) or ms for raster units")
    ap.add_argument("--ecmwf-units", default="kt", help="kt (default) or ms for raster units")
    ap.add_argument("--cmc-units", default="kt", help="kt (default) or ms for raster units")
    ap.add_argument("--icon-units", default="kt", help="kt (default) or ms for raster units")

    args = ap.parse_args()

    if not _USE_RASTERIO and not _USE_GDAL:
        print("ERROR: Need rasterio or GDAL available to sample GeoTIFFs.", file=sys.stderr)
        return 2

    # ---------- input obs ----------
    in_path = Path(args.in_path) if args.in_path else None

    # If not specified, try common candidates
    if in_path is None:
        candidates = [
            Path("data/raob/obs_latest.json"),
            Path("data/raob/obs_latest.csv"),
            Path("data/raob/latest_obs.csv"),
            Path("data/raob/obs.csv"),
            Path("data/raob/obs.json"),
        ]
        in_path = next((p for p in candidates if p.exists()), None)

    if in_path is None or not in_path.exists():
        print("ERROR: No input obs file found. Use --in <file.csv|file.json>", file=sys.stderr)
        return 2

    valid_guess, obs_rows = detect_and_load_obs(in_path)
    if not obs_rows:
        print(f"ERROR: No valid obs rows parsed from {in_path}", file=sys.stderr)
        return 2

    valid_utc = (args.valid_utc or valid_guess or obs_rows[0].valid_utc or "").strip()
    if not valid_utc:
        # last resort: current hour Z
        valid_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00Z")

    # ---------- resolve rasters ----------
    # Patterns are intentionally broad; newest match wins.
    gfs_path = _resolve_candidate_paths(args.gfs_tif, [
        "**/*gfs*250*speed*.tif",
        "**/*gfs*250*.tif",
        "**/output_250mb*.tif",
        "**/gfs*250mb*.tif",
    ])
    ecmwf_path = _resolve_candidate_paths(args.ecmwf_tif, [
        "**/*ecmwf*250*speed*.tif",
        "**/*ecmwf*250*.tif",
        "**/ecmwf*250mb*.tif",
    ])
    cmc_path = _resolve_candidate_paths(args.cmc_tif, [
        "**/*cmc*250*speed*.tif",
        "**/*gem*250*speed*.tif",
        "**/*cmc*250*.tif",
        "**/*gem*250*.tif",
    ])
    icon_path = _resolve_candidate_paths(args.icon_tif, [
        "**/*icon*250*speed*.tif",
        "**/*icon*250*.tif",
    ])

    def _safe_open(p: Optional[Path], units: str, label: str) -> Optional[RasterInfo]:
        if p is None:
            print(f"[WARN] {label}: no GeoTIFF found -> will write null", file=sys.stderr)
            return None
        try:
            info = open_raster(p, units)
            print(f"[OK] {label}: {p}  (lon {info.lon_min}..{info.lon_max}, lat {info.lat_min}..{info.lat_max}, units={info.units}, backend={info.backend})")
            return info
        except Exception as e:
            print(f"[WARN] {label}: failed to open {p}: {e} -> will write null", file=sys.stderr)
            return None

    model_rasters = {
        "GFS": _safe_open(gfs_path, args.gfs_units, "GFS"),
        "ECMWF": _safe_open(ecmwf_path, args.ecmwf_units, "ECMWF"),
        "CMC": _safe_open(cmc_path, args.cmc_units, "CMC"),
        "ICON": _safe_open(icon_path, args.icon_units, "ICON"),
    }

    out_path = Path(args.out_path)
    build_latest_json(obs_rows, valid_utc, model_rasters, out_path)
    print(f"[DONE] wrote {out_path} (rows={len(obs_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
