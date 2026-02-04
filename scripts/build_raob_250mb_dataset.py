# scripts/build_raob_250mb_dataset.py
#!/usr/bin/env python3
"""
scripts/build_raob_250mb_dataset.py

- Reads RAOB obs (CSV/JSON) for all stations.
- Samples 250mb SPEED GeoTIFFs for GFS / ECMWF / CMC / ICON.
- Writes data/raob/latest.json in the format the web map expects.
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
    import numpy as np  # type: ignore
    import rasterio  # type: ignore

    _USE_RASTERIO = True
except Exception:
    try:
        import numpy as np  # type: ignore
        from osgeo import gdal  # type: ignore

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


def guess_valid_utc(payload: Any) -> str:
    if isinstance(payload, dict):
        meta = payload.get("meta") or {}
        for k in ("valid_utc", "validUtc", "valid", "time"):
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for k in ("valid_utc", "validUtc", "valid", "time"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def load_obs(path: Path) -> Tuple[str, List[ObsRow]]:
    if not path.exists():
        raise FileNotFoundError(path)

    valid_guess = ""
    rows_raw: List[Dict[str, Any]] = []

    if path.suffix.lower() == ".csv":
        rows_raw = read_rows_from_csv(path)
    else:
        payload, rows_raw = read_rows_from_json(path)
        valid_guess = guess_valid_utc(payload)

    out: List[ObsRow] = []
    for r in rows_raw:
        station = normalize_station_id(
            r.get("station")
            or r.get("id")
            or r.get("identifier")
            or r.get("icao")
            or r.get("wmo")
        )
        if not station:
            continue

        name = str(
            r.get("name")
            or r.get("location")
            or r.get("locName")
            or r.get("location_name")
            or ""
        ).strip()

        lat = to_float_or_none(r.get("lat") or r.get("latitude"))
        lon = to_float_or_none(r.get("lon") or r.get("longitude"))
        obs = to_float_or_none(
            r.get("obs")
            or r.get("raob_speed_kt")
            or r.get("obs_kt")
            or r.get("raob")
        )
        vutc = str(
            r.get("valid_utc") or r.get("validUtc") or r.get("valid") or ""
        ).strip()

        if lat is None or lon is None or obs is None:
            continue

        out.append(
            ObsRow(
                station=station,
                name=name,
                lat=lat,
                lon=lon,
                obs_kt=obs,
                valid_utc=vutc,
            )
        )

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
    units: str  # "kt" or "ms"


def _resolve_candidate(explicit: Optional[str], patterns: List[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for pat in patterns:
        hits = list(Path(".").glob(pat))
        hits = [h for h in hits if h.is_file()]
        if hits:
            hits.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return hits[0]
    return None


def open_raster(path: Path, units: str) -> RasterInfo:
    units = (units or "kt").lower()
    units_norm = "ms" if units in ("ms", "m/s", "mps") else "kt"

    if _USE_RASTERIO:
        import rasterio

        with rasterio.open(path) as ds:
            b = ds.bounds
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
        from osgeo import gdal  # type: ignore

        ds = gdal.Open(str(path))
        if ds is None:
            raise RuntimeError(f"GDAL could not open: {path}")
        gt = ds.GetGeoTransform()
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

    raise RuntimeError("No raster backend available (need rasterio or GDAL).")


def _adjust_lon(lon: float, info: RasterInfo) -> float:
    if info.lon_min >= 0 and lon < 0:
        lon = lon % 360.0
    if info.lon_max <= 180 and lon > 180:
        lon = ((lon + 180) % 360) - 180
    return lon


def sample_points(info: RasterInfo, points: List[Tuple[float, float]]) -> List[Optional[float]]:
    vals: List[Optional[float]] = [None] * len(points)

    if info.backend == "rasterio":
        import rasterio
        import numpy as np

        with rasterio.open(info.path) as ds:
            coords = []
            for lat, lon in points:
                lon2 = _adjust_lon(lon, info)
                coords.append((float(lon2), float(lat)))

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
        from osgeo import gdal  # type: ignore
        import numpy as np

        ds = gdal.Open(str(info.path))
        band = ds.GetRasterBand(1)
        gt = ds.GetGeoTransform()
        success, inv_gt = gdal.InvGeoTransform(gt)
        if not success:
            return vals

        for i, (lat, lon) in enumerate(points):
            lon2 = _adjust_lon(lon, info)
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
            if not np.isfinite(v):
                vals[i] = None
                continue

            if info.units == "ms":
                v *= MS_TO_KT
            vals[i] = float(v)
        return vals

    return vals


def build_json(
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
        sampled[model] = sample_points(rinfo, coords)

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
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="", help="Input obs CSV/JSON (station, name, lat, lon, obs)")
    ap.add_argument("--out", dest="out_path", default="data/raob/latest.json", help="Output JSON path")
    ap.add_argument("--valid-utc", dest="valid_utc", default="", help="Valid UTC (YYYY-MM-DDTHHZ)")

    ap.add_argument("--gfs-tif", default="", help="GFS speed GeoTIFF")
    ap.add_argument("--ecmwf-tif", default="", help="ECMWF speed GeoTIFF")
    ap.add_argument("--cmc-tif", default="", help="CMC speed GeoTIFF")
    ap.add_argument("--icon-tif", default="", help="ICON speed GeoTIFF")

    ap.add_argument("--gfs-units", default="kt", help="kt (default) or ms")
    ap.add_argument("--ecmwf-units", default="kt", help="kt (default) or ms")
    ap.add_argument("--cmc-units", default="kt", help="kt (default) or ms")
    ap.add_argument("--icon-units", default="kt", help="kt (default) or ms")

    args = ap.parse_args()

    if not (_USE_RASTERIO or _USE_GDAL):
        print("ERROR: need rasterio or GDAL to sample GeoTIFFs", file=sys.stderr)
        return 2

    # obs file
    in_path = Path(args.in_path) if args.in_path else None
    if in_path is None:
        for cand in [
            Path("data/raob/obs_latest.csv"),
            Path("data/raob/obs_latest.json"),
            Path("data/raob/latest_obs.csv"),
        ]:
            if cand.exists():
                in_path = cand
                break

    if in_path is None or not in_path.exists():
        print("ERROR: no obs file found, use --in", file=sys.stderr)
        return 2

    valid_guess, obs_rows = load_obs(in_path)
    if not obs_rows:
        print("ERROR: parsed 0 obs rows", file=sys.stderr)
        return 2

    # valid time: env VALID_UTC (YYYYMMDDHH) -> ISO, or args / guess
    v_env = os.environ.get("VALID_UTC", "").strip()
    v_arg = args.valid_utc.strip()
    valid_utc = ""

    if v_arg:
        valid_utc = v_arg
    elif v_env and len(v_env) == 10:
        valid_utc = f"{v_env[0:4]}-{v_env[4:6]}-{v_env[6:8]}T{v_env[8:10]}:00Z"
    elif valid_guess:
        valid_utc = valid_guess
    elif obs_rows[0].valid_utc:
        valid_utc = obs_rows[0].valid_utc
    else:
        valid_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00Z")

    def resolve(explicit: str, pats: List[str]) -> Optional[Path]:
        return _resolve_candidate(explicit or None, pats)

    gfs_path = resolve(
        args.gfs_tif,
        ["output_250mb_speed.tif", "**/*gfs*250*speed*.tif", "**/output_250mb_speed*.tif"],
    )
    ecmwf_path = resolve(
        args.ecmwf_tif,
        ["**/*ecmwf*250*speed*.tif"],
    )
    cmc_path = resolve(
        args.cmc_tif,
        ["**/*cmc*250*speed*.tif", "**/*gem*250*speed*.tif"],
    )
    icon_path = resolve(
        args.icon_tif,
        ["**/*icon*250*speed*.tif"],
    )

    def safe_open(p: Optional[Path], units: str, label: str) -> Optional[RasterInfo]:
        if p is None:
            print(f"[WARN] {label}: no GeoTIFF found -> will write null", file=sys.stderr)
            return None
        try:
            info = open_raster(p, units)
            print(
                f"[OK] {label}: {p} (lon {info.lon_min}..{info.lon_max}, lat {info.lat_min}..{info.lat_max}, units={info.units}, backend={info.backend})"
            )
            return info
        except Exception as e:
            print(f"[WARN] {label}: failed to open {p}: {e} -> will write null", file=sys.stderr)
            return None

    model_rasters = {
        "GFS": safe_open(gfs_path, args.gfs_units, "GFS"),
        "ECMWF": safe_open(ecmwf_path, args.ecmwf_units, "ECMWF"),
        "CMC": safe_open(cmc_path, args.cmc_units, "CMC"),
        "ICON": safe_open(icon_path, args.icon_units, "ICON"),
    }

    out_path = Path(args.out_path)
    build_json(obs_rows, valid_utc, model_rasters, out_path)
    print(f"[DONE] wrote {out_path} (rows={len(obs_rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
