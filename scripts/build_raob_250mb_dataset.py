#!/usr/bin/env python3
"""
Build global 250mb RAOB vs model dataset for ONE valid time (VALID_UTC).

Apples-to-apples rules:
- We ONLY accept RAOBs valid at VALID_UTC (±15 min for IEM; exact hour for UWyo TEXT:LIST ddhh)
- Model values are for the SAME valid time.
- GFS values are sampled from the SAME GRIB used to make the tiles: data/gfs.grib2

Outputs:
- data/raob/latest.json
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests
import xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed

TARGET_MB = 250.0
KT_PER_MS = 1.94384
MAX_IEM_OFFSET_MIN = 15

IEM_RAOB_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/raob.py"
UWYO_URL = "http://weather.uwyo.edu/cgi-bin/sounding"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

OPEN_METEO_MODELS = {
    "ECMWF": "ecmwf_ifs04",
    "CMC": "gem_global",
    "ICON": "icon_seamless",
}

UA = {
    "User-Agent": "Mozilla/5.0 (GitHubActions; RAOB/250mb) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"
}


@dataclass(frozen=True)
class Station:
    key: str
    name: str
    region: str
    id_raw: str
    lat: float
    lon: float
    wmo5: Optional[str]  # for UWyo STNM if available


# ---------------- helpers ----------------

def _norm_key(s: str) -> str:
    return (s or "").strip().lower()

def _first_present(row: dict, keys: List[str]) -> Optional[str]:
    lk = {_norm_key(k): k for k in row.keys()}
    for want in keys:
        actual = lk.get(_norm_key(want))
        if actual is not None:
            v = row.get(actual)
            if v is None:
                continue
            v = str(v).strip()
            if v != "":
                return v
    return None

def _parse_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(str(x).strip())
    except Exception:
        return None

def wmo5_from_any(id_raw: str, igra_id: Optional[str], explicit: Optional[str]) -> Optional[str]:
    # explicit WMO fields
    for src in [explicit, id_raw, igra_id]:
        if not src:
            continue
        s = str(src).strip()
        # exact 5-digit
        if re.fullmatch(r"\d{5}", s):
            return s
        # IGRA ID often ends with 5 digits (ex: USM00072233)
        m = re.search(r"(\d{5})\s*$", s)
        if m:
            return m.group(1)
    return None

def is_iem_coverage(lat: float, lon: float) -> bool:
    # same bounding box you used in Apps Script
    return (lat >= 10 and lat <= 75) and (lon >= -170 and lon <= -40)

def guess_uwyo_region(lat: float, lon: float) -> str:
    # ported from your Apps Script logic
    if lat <= -60:
        return "ant"
    if lat >= 60 and (-20 < lon < 180):
        return "np"
    if lon <= -20 and lon >= -170:
        return "naconf" if (lat >= 10) else "samer"
    if -20 < lon <= 60:
        if lat >= 35:
            return "europe"
        if lat >= 10 and lon >= 20:
            return "mideast"
        return "africa"
    if 60 < lon < 180:
        if lat < -10 and lon > 140:
            return "nz"
        if lat < 15 and (lon > 150 or lon < 80):
            return "pac"
        return "seasia"
    return "seasia"

def estimate_at_250(levels: List[Tuple[float, float]]) -> Optional[float]:
    # exact-ish
    for p, spd in levels:
        if abs(p - TARGET_MB) <= 0.5:
            return spd

    above = [x for x in levels if x[0] < TARGET_MB]
    below = [x for x in levels if x[0] > TARGET_MB]
    if above and below:
        # closest on each side in pressure space
        p1, s1 = max(above, key=lambda x: x[0])  # highest pressure below target? (closest above in altitude)
        p2, s2 = min(below, key=lambda x: x[0])
        # log-pressure interpolation
        w = (math.log(TARGET_MB) - math.log(p1)) / (math.log(p2) - math.log(p1))
        return s1 + (s2 - s1) * w

    return None


# ---------------- station loading ----------------

def load_stations(stations_dir: str) -> List[Station]:
    csv_files = sorted(
        f for f in os.listdir(stations_dir)
        if f.lower().endswith(".csv")
    )
    if not csv_files:
        raise RuntimeError(f"No CSVs found in {stations_dir}")

    stations: List[Station] = []

    for fname in csv_files:
        path = os.path.join(stations_dir, fname)
        region_default = os.path.splitext(fname)[0]

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lat = _parse_float(_first_present(row, ["Latitude", "Lat"]))
                lon = _parse_float(_first_present(row, ["Longitude", "Lon", "Long"]))
                if lat is None or lon is None:
                    continue

                name = _first_present(row, ["Location Name", "Name", "Station Name", "Station"]) or ""
                region = _first_present(row, ["Region", "Continent"]) or region_default

                id_raw = _first_present(row, ["Identifier", "ID", "Station", "ICAO", "Site", "site_id"]) or ""
                igra_id = _first_present(row, ["igra_id", "IGRA_ID"])  # if present
                wmo_explicit = _first_present(row, ["WY_STNM", "wy_stnm", "WMO", "wmo", "wmo_id", "WMO_ID"])

                wmo5 = wmo5_from_any(id_raw=id_raw, igra_id=igra_id, explicit=wmo_explicit)

                # if no id but we do have WMO, use that as id_raw
                if not id_raw and wmo5:
                    id_raw = wmo5

                if not id_raw and not wmo5:
                    continue

                sid = id_raw.strip().upper()
                key = f"{sid}|{lat:.4f}|{lon:.4f}"

                stations.append(Station(
                    key=key,
                    name=(name.strip() or sid),
                    region=region.strip(),
                    id_raw=sid,
                    lat=float(lat),
                    lon=float(lon),
                    wmo5=wmo5
                ))

    # de-dupe by key
    uniq: Dict[str, Station] = {}
    for s in stations:
        uniq[s.key] = s

    out = list(uniq.values())
    print(f"Loaded {len(out)} stations from /stations")
    return out


# ---------------- RAOB fetchers ----------------

def iem_station_candidates(id_raw: str) -> List[str]:
    s = (id_raw or "").strip().upper()
    out = []
    if s:
        out.append(s)
    # common fix: KLIX -> LIX, CYYZ -> YYZ
    if len(s) == 4 and s[0] in ("K", "C"):
        out.append(s[1:])
    # sometimes already 3-letter
    if len(s) == 3:
        out.append(s)
    # WMO digits sometimes work depending on endpoint
    if re.fullmatch(r"\d{5}", s):
        out.append(s)
    # de-dupe preserving order
    seen = set()
    dedup = []
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup

def fetch_iem_raob_250(id_raw: str, target_dt: datetime) -> Optional[Tuple[datetime, float, str]]:
    sts = (target_dt - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%MZ")
    ets = (target_dt + timedelta(hours=3)).strftime("%Y-%m-%dT%H:%MZ")

    for cand in iem_station_candidates(id_raw):
        url = f"{IEM_RAOB_URL}?station={requests.utils.quote(cand)}&sts={requests.utils.quote(sts)}&ets={requests.utils.quote(ets)}"
        try:
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code != 200 or not r.text.strip():
                continue

            rows = list(csv.reader(r.text.splitlines()))
            if len(rows) < 2:
                continue

            header = [h.strip() for h in rows[0]]
            def idx(name: str) -> int:
                for i, h in enumerate(header):
                    if h.strip().lower() == name.lower():
                        return i
                return -1

            i_valid = idx("validUTC")
            i_p = idx("pressure_mb")
            i_spd = idx("speed_kts")
            if i_spd < 0:
                i_spd = idx("sknt")

            if i_valid < 0 or i_p < 0 or i_spd < 0:
                continue

            groups: Dict[datetime, List[Tuple[float, float]]] = {}

            for row in rows[1:]:
                if len(row) <= max(i_valid, i_p, i_spd):
                    continue
                valid = row[i_valid].strip()
                m = re.match(r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})", valid)
                if not m:
                    continue
                dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), tzinfo=timezone.utc)

                try:
                    p = float(row[i_p])
                    spd = float(row[i_spd])
                except Exception:
                    continue

                groups.setdefault(dt, []).append((p, spd))

            # best match to target
            best_dt = None
            best_diff = None
            for dt in groups.keys():
                diff = abs((dt - target_dt).total_seconds()) / 60.0
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_dt = dt

            if best_dt is None or best_diff is None or best_diff > MAX_IEM_OFFSET_MIN:
                continue

            est = estimate_at_250(groups[best_dt])
            if est is None:
                continue

            return (best_dt, est, f"IEM:{cand}")

        except Exception:
            continue

    return None

def parse_uwyo_text_list_levels(html: str) -> List[Tuple[float, float]]:
    m = re.search(r"<pre[^>]*>([\s\S]*?)</pre>", html, re.IGNORECASE)
    if not m:
        return []
    pre = m.group(1)
    pre = (pre
           .replace("&nbsp;", " ")
           .replace("&lt;", "<")
           .replace("&gt;", ">")
           .replace("&amp;", "&"))
    lines = pre.splitlines()

    start = -1
    for i, line in enumerate(lines):
        if re.match(r"^\s*PRES\s+", line, re.IGNORECASE):
            start = i + 1
            break
    if start < 0:
        return []

    levels: List[Tuple[float, float]] = []
    for line in lines[start:]:
        if re.match(r"^\s*(Station|Observations|Index|Showalter|Lifted)", line, re.IGNORECASE):
            break
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        try:
            p = float(parts[0])
            sknt = float(parts[7])  # same column you used in Apps Script
        except Exception:
            continue
        levels.append((p, sknt))
    return levels

def fetch_uwyo_raob_250(stnm: str, lat: float, lon: float, target_dt: datetime) -> Optional[Tuple[datetime, float, str]]:
    region = guess_uwyo_region(lat, lon)
    year = target_dt.strftime("%Y")
    month = target_dt.strftime("%m")
    day = target_dt.strftime("%d")
    hour = target_dt.strftime("%H")
    ddhh = f"{day}{hour}"

    url = (
        f"{UWYO_URL}?region={requests.utils.quote(region)}"
        f"&TYPE=TEXT:LIST&YEAR={year}&MONTH={month}&FROM={ddhh}&TO={ddhh}"
        f"&STNM={requests.utils.quote(stnm)}"
    )

    try:
        r = requests.get(url, headers=UA, timeout=60, allow_redirects=True)
        if r.status_code != 200:
            return None
        levels = parse_uwyo_text_list_levels(r.text)
        if not levels:
            return None
        est = estimate_at_250(levels)
        if est is None:
            return None
        # UWyo is exactly ddhh; treat as exact
        return (target_dt, est, f"UWyo:{stnm}:{region}")
    except Exception:
        return None

def fetch_raob_250_for_station(st: Station, target_dt: datetime) -> Optional[dict]:
    # UWyo first if we have WMO
    if st.wmo5:
        uw = fetch_uwyo_raob_250(st.wmo5, st.lat, st.lon, target_dt)
        if uw:
            dt, obs, src = uw
            return {"dt": dt, "obs": obs, "src": src}

    # IEM fallback (mostly NA)
    if is_iem_coverage(st.lat, st.lon):
        iem = fetch_iem_raob_250(st.id_raw, target_dt)
        if iem:
            dt, obs, src = iem
            return {"dt": dt, "obs": obs, "src": src}

    # one last attempt even outside box (cheap)
    iem = fetch_iem_raob_250(st.id_raw, target_dt)
    if iem:
        dt, obs, src = iem
        return {"dt": dt, "obs": obs, "src": src}

    return None


# ---------------- model fetchers ----------------

def fetch_open_meteo_models(lat: float, lon: float, target_dt: datetime) -> Dict[str, Optional[float]]:
    date = target_dt.strftime("%Y-%m-%d")
    tkey = target_dt.strftime("%Y-%m-%dT%H:00")

    url = (
        f"{OPEN_METEO_URL}?latitude={lat}&longitude={lon}"
        f"&start_date={date}&end_date={date}"
        f"&hourly=wind_speed_250hPa&wind_speed_unit=kn"
        f"&models=gfs_seamless,ecmwf_ifs04,gem_global,icon_seamless"
        f"&timezone=UTC"
    )

    out: Dict[str, Optional[float]] = {"ECMWF": None, "CMC": None, "ICON": None}

    try:
        r = requests.get(url, headers=UA, timeout=60)
        if r.status_code != 200:
            return out
        j = r.json()
        hourly = j.get("hourly") or {}
        times = hourly.get("time") or []
        if tkey not in times:
            return out
        idx = times.index(tkey)
        for label, suffix in OPEN_METEO_MODELS.items():
            arr = hourly.get(f"wind_speed_250hPa_{suffix}")
            if isinstance(arr, list) and idx < len(arr):
                out[label] = arr[idx]
    except Exception:
        return out

    return out


def sample_gfs_from_grib(grib_path: str, stations: List[Station]) -> Dict[str, Optional[float]]:
    # Force isobaricInhPa to avoid cfgrib multiple typeOfLevel
    ds_u = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "shortName": "u"},
            "indexpath": ""
        },
    )
    ds_v = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa", "shortName": "v"},
            "indexpath": ""
        },
    )

    u = ds_u["u"].sel(isobaricInhPa=250)
    v = ds_v["v"].sel(isobaricInhPa=250)

    out: Dict[str, Optional[float]] = {}
    for st in stations:
        try:
            lon = st.lon if st.lon >= 0 else st.lon + 360.0
            uu = float(u.interp(latitude=st.lat, longitude=lon))
            vv = float(v.interp(latitude=st.lat, longitude=lon))
            out[st.key] = math.sqrt(uu * uu + vv * vv) * KT_PER_MS
        except Exception:
            out[st.key] = None
    return out


# ---------------- main ----------------

def main():
    valid = os.environ.get("VALID_UTC", "").strip()
    if not valid:
        raise RuntimeError("VALID_UTC env var is missing/empty")
    target_dt = datetime.strptime(valid, "%Y%m%d%H").replace(tzinfo=timezone.utc)

    stations = load_stations("stations")
    gfs = sample_gfs_from_grib("data/gfs.grib2", stations)

    # 1) fetch obs first (only compute models where obs exists)
    obs_results: Dict[str, dict] = {}
    misses = 0

    max_workers = int(os.environ.get("RAOB_WORKERS", "8"))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_raob_250_for_station, st, target_dt): st for st in stations}
        for fut in as_completed(futs):
            st = futs[fut]
            res = fut.result()
            if res is None:
                misses += 1
                continue
            obs_results[st.key] = res

    print(f"RAOB hits: {len(obs_results)} / {len(stations)} (misses {misses})")

    # 2) fetch models only for stations that have obs
    rows: List[dict] = []

    def delta(model_val: Optional[float], obs_val: float) -> Optional[float]:
        if model_val is None:
            return None
        return float(model_val) - float(obs_val)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {}
        key_to_station = {st.key: st for st in stations}

        for key, obsinfo in obs_results.items():
            st = key_to_station[key]
            futs[ex.submit(fetch_open_meteo_models, st.lat, st.lon, target_dt)] = (st, obsinfo)

        for fut in as_completed(futs):
            st, obsinfo = futs[fut]
            other = fut.result()

            obs = float(obsinfo["obs"])
            src = obsinfo["src"]
            obs_dt = obsinfo["dt"]

            gfs_val = gfs.get(st.key)

            rows.append({
                "name": st.name,
                "region": st.region,
                "id": st.id_raw,
                "wmo5": st.wmo5,
                "lat": st.lat,
                "lon": st.lon,
                "valid_utc": obs_dt.strftime("%Y-%m-%dT%H:%MZ"),
                "obs_src": src,
                "obs": obs,
                "models": {
                    "GFS":   {"speed": gfs_val,         "delta": delta(gfs_val, obs)},
                    "ECMWF": {"speed": other["ECMWF"],  "delta": delta(other["ECMWF"], obs)},
                    "CMC":   {"speed": other["CMC"],    "delta": delta(other["CMC"], obs)},
                    "ICON":  {"speed": other["ICON"],   "delta": delta(other["ICON"], obs)},
                }
            })

    out = {
        "meta": {
            "valid_utc": target_dt.strftime("%Y-%m-%dT%H:%MZ"),
            "generated_utc": datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "station_count": len(stations),
            "raob_point_count": len(rows),
        },
        "rows": rows
    }

    os.makedirs("data/raob", exist_ok=True)
    with open("data/raob/latest.json", "w", encoding="utf-8") as f:
        json.dump(out, f)

    print(f"Wrote {len(rows)} RAOB points to data/raob/latest.json")


if __name__ == "__main__":
    main()
