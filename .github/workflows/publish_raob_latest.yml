#!/usr/bin/env python3
"""
scripts/publish_raob_latest.py

Atomic publisher:
- Finds the newest cycle under data/raob/staging/<cycle> that has ALL required model partials.
- Merges them into data/raob/latest.json
- Archives combined JSON to data/history/<cycle>.json
- Promotes tiles for that cycle -> tiles/<model>/latest/

This prevents the website from ever showing a "partial model" cycle.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ALL_MODELS = ["GFS", "ECMWF", "CMC", "ICON"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_min(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def safe_rmtree(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)


def copytree_into(src: Path, dst: Path) -> None:
    safe_rmtree(dst)
    shutil.copytree(src, dst)


def normalize_station_id(x: Any) -> str:
    return str(x or "").strip().upper()


def newest_complete_cycle(
    staging_root: Path,
    required_models: List[str],
    tiles_map: Dict[str, Path],
) -> Optional[str]:
    if not staging_root.exists():
        return None

    cycles = sorted([p.name for p in staging_root.iterdir() if p.is_dir() and p.name.isdigit()], reverse=True)
    for cyc in cycles:
        # required partials must exist
        ok = True
        for m in required_models:
            if not (staging_root / cyc / f"{m}.json").exists():
                ok = False
                break
        if not ok:
            continue

        # required tiles must exist
        for m in required_models:
            base = tiles_map.get(m)
            if not base:
                ok = False
                break
            if not (base / cyc).exists():
                ok = False
                break
        if not ok:
            continue

        return cyc

    return None


def merge_partials_for_cycle(
    staging_root: Path,
    cycle: str,
    required_models: List[str],
) -> Dict[str, Any]:
    # Use the first required model as the "base" rows (station list + obs + lat/lon/name)
    base_model = required_models[0]
    base_path = staging_root / cycle / f"{base_model}.json"
    base_payload = read_json(base_path)

    valid_utc = base_payload.get("meta", {}).get("valid_utc", "")
    rows = base_payload.get("rows", [])

    # station_id -> row object
    out_map: Dict[str, Dict[str, Any]] = {}

    def ensure_models_dict(row_obj: Dict[str, Any]) -> None:
        models = row_obj.setdefault("models", {})
        for m in ALL_MODELS:
            models.setdefault(m, {"speed": None, "delta": None})

    for r in rows:
        sid = normalize_station_id(r.get("id") or r.get("station"))
        if not sid:
            continue

        row_obj = {
            "name": r.get("name") or "",
            "id": sid,
            "lat": r.get("lat"),
            "lon": r.get("lon"),
            "valid_utc": valid_utc,
            "obs": r.get("obs"),
            "models": {},
        }
        ensure_models_dict(row_obj)

        # base file only has base_model populated; copy it
        mobj = (r.get("models") or {}).get(base_model) or {}
        row_obj["models"][base_model] = {
            "speed": mobj.get("speed"),
            "delta": mobj.get("delta"),
        }
        out_map[sid] = row_obj

    # Merge remaining required models
    for m in required_models[1:]:
        p = staging_root / cycle / f"{m}.json"
        payload = read_json(p)
        v2 = payload.get("meta", {}).get("valid_utc", "")
        if v2 != valid_utc:
            # This *shouldn't* happen, but if it does, ignore this model for safety.
            print(f"[WARN] {m} valid_utc mismatch ({v2} != {valid_utc}); skipping merge for this model")
            continue

        for r in payload.get("rows", []):
            sid = normalize_station_id(r.get("id") or r.get("station"))
            if not sid:
                continue
            if sid not in out_map:
                # if station appears here but not in base, create it
                row_obj = {
                    "name": r.get("name") or "",
                    "id": sid,
                    "lat": r.get("lat"),
                    "lon": r.get("lon"),
                    "valid_utc": valid_utc,
                    "obs": r.get("obs"),
                    "models": {},
                }
                ensure_models_dict(row_obj)
                out_map[sid] = row_obj

            mobj = (r.get("models") or {}).get(m) or {}
            out_map[sid]["models"][m] = {
                "speed": mobj.get("speed"),
                "delta": mobj.get("delta"),
            }

    # models_present
    models_present = {m: False for m in ALL_MODELS}
    for m in ALL_MODELS:
        for row in out_map.values():
            spd = (row.get("models") or {}).get(m, {}).get("speed")
            if spd is not None:
                models_present[m] = True
                break

    merged_rows = list(out_map.values())

    return {
        "meta": {
            "cycle": cycle,
            "valid_utc": valid_utc,
            "generated_utc": utc_now_iso(),
            "station_count": len(merged_rows),
            "models_present": models_present,
        },
        "rows": merged_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging-root", default="data/raob/staging")
    ap.add_argument("--required-models", default="GFS,ECMWF", help="Comma-separated required models for publishing")
    ap.add_argument("--out-latest", default="data/raob/latest.json")
    ap.add_argument("--history-dir", default="data/history")
    ap.add_argument("--tiles-map", default="GFS=tiles/250mb,ECMWF=tiles/ecmwf", help="Comma-separated MODEL=PATH pairs")
    args = ap.parse_args()

    staging_root = Path(args.staging_root)
    out_latest = Path(args.out_latest)
    history_dir = Path(args.history_dir)

    required_models = [m.strip().upper() for m in (args.required_models or "").split(",") if m.strip()]
    if not required_models:
        print("ERROR: required-models is empty", file=sys.stderr)
        return 2
    for m in required_models:
        if m not in ALL_MODELS:
            print(f"ERROR: unknown model in required-models: {m}", file=sys.stderr)
            return 2

    tiles_map: Dict[str, Path] = {}
    for part in (args.tiles_map or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            print(f"ERROR: bad tiles-map segment: {part}", file=sys.stderr)
            return 2
        m, p = part.split("=", 1)
        tiles_map[m.strip().upper()] = Path(p.strip())

    cycle = newest_complete_cycle(staging_root, required_models, tiles_map)
    if not cycle:
        print("[INFO] No complete cycle available yet. Nothing to publish.")
        return 0

    # If already published, don’t do anything
    if out_latest.exists():
        try:
            old = read_json(out_latest)
            old_cycle = old.get("meta", {}).get("cycle", "")
            if old_cycle == cycle:
                print(f"[INFO] latest already on cycle {cycle}. Nothing to do.")
                return 0
        except Exception:
            pass

    merged = merge_partials_for_cycle(staging_root, cycle, required_models)

    # Write latest + archive
    write_json_min(out_latest, merged)
    archive_path = history_dir / f"{cycle}.json"
    write_json_min(archive_path, merged)

    # Promote tiles to /latest for required models
    for m in required_models:
        base = tiles_map[m]
        src = base / cycle
        dst = base / "latest"
        if not src.exists():
            print(f"[WARN] Missing tiles for {m} at {src}; skipping tile promotion")
            continue
        copytree_into(src, dst)
        # Optional marker file
        (dst / "last_update.txt").write_text(utc_now_iso() + "\n", encoding="utf-8")

    print(f"[OK] Published cycle {cycle} (required={required_models})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
