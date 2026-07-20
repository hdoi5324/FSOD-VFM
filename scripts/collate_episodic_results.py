#!/usr/bin/env python3
"""Collate FSOD-VFM episodic results across k-shots and seeds.

Scans a directory of exported episode packages (each a subdir with
``provenance.json`` and a work dir produced by
``scripts/run_episodic_coco_package.py``), reads per-episode COCO metrics
(``coco_eval_stats_.txt``) and timing (``timing.json``), and writes:

  * ``episodes.csv``          - one row per (source_method, k_shot, seed) episode
  * ``summary_by_kshot.csv``  - mean/std over seeds, grouped by source_method + k_shot
  * ``summary.json``          - the same aggregate as nested JSON

This mirrors the NTTT / fsod_eval ``aggregate_episode_metrics`` style
(mean/std with sample std, ddof=1) so FSOD-VFM numbers are directly
comparable to SAM3 and no-time-to-train.

Example::

    python scripts/collate_episodic_results.py \\
      --packages-dir /path/to/episode_packages/13393 \\
      --out-dir      /path/to/episode_packages/13393/fsodvfm_results \\
      --work-subdir  fsodvfm_work
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Order matches the names written by run_episodic_coco_package.py
# (same labels as NTTT's coco_ref_dataset.evaluate).
STAT_NAME_TO_KEY = {
    "AP IoU=0.50:0.95": "AP",
    "AP IoU=0.50": "AP50",
    "AP IoU=0.75": "AP75",
    "AP small": "AP_small",
    "AP medium": "AP_medium",
    "AP large": "AP_large",
    "AR maxDets=1": "AR1",
    "AR maxDets=10": "AR10",
    "AR maxDets=100": "AR100",
    "AR small": "AR_small",
    "AR medium": "AR_medium",
    "AR large": "AR_large",
}

CSV_STAT_KEYS = ("AP", "AP50", "AP75", "AR100")
IOU_TYPES = ("bbox", "segm")
TIMING_KEYS = (
    "fill_memory_s",
    "postprocess_memory_s",
    "test_s",
    "total_s",
    "test_s_per_image",
)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def parse_coco_eval_stats(path: Path) -> Dict[str, Dict[str, float]]:
    """Parse coco_eval_stats_.txt into {"bbox": {...}, "segm": {...}}."""
    out: Dict[str, Dict[str, float]] = {"bbox": {}, "segm": {}}
    section: Optional[str] = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "BBOX RESULTS" in line:
            section = "bbox"
            continue
        if "SEGM RESULTS" in line:
            section = "segm"
            continue
        if section is None or ": " not in line:
            continue
        name, val = line.rsplit(": ", 1)
        key = STAT_NAME_TO_KEY.get(name.strip())
        if key is None:
            continue
        try:
            out[section][key] = float(val)
        except ValueError:
            pass
    return out


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_episodes(
    packages_dir: Path,
    *,
    work_subdir: str = "fsodvfm_work",
    k_shots: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    for prov_path in sorted(packages_dir.glob("*/provenance.json")):
        pkg = prov_path.parent
        prov = _load_json(prov_path)
        k = prov.get("k_shot")
        seed = prov.get("seed")
        if k_shots is not None and k not in k_shots:
            continue
        if seeds is not None and seed not in seeds:
            continue
        work = pkg / work_subdir
        stats_file = work / "coco_eval_stats_.txt"
        timing_file = work / "timing.json"
        row: Dict[str, Any] = {
            "package": pkg.name,
            "package_dir": str(pkg),
            "source_method": prov.get("method"),
            "k_shot": k,
            "seed": seed,
            "category_num": prov.get("category_num"),
            "cat_names_csv": prov.get("cat_names_csv"),
            "n_test_images": prov.get("n_test_images"),
            "has_stats": stats_file.is_file(),
            "has_timing": timing_file.is_file(),
            "stats": parse_coco_eval_stats(stats_file) if stats_file.is_file() else None,
            "timing": _load_json(timing_file) if timing_file.is_file() else None,
        }
        episodes.append(row)
    return episodes


def write_episodes_csv(episodes: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "package",
        "source_method",
        "k_shot",
        "seed",
        "category_num",
        "n_test_images",
    ]
    for iou in IOU_TYPES:
        for k in CSV_STAT_KEYS:
            fieldnames.append(f"{iou}_{k}")
    fieldnames.extend(TIMING_KEYS)
    fieldnames.append("status")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in episodes:
            row: Dict[str, Any] = {
                "package": ep["package"],
                "source_method": ep["source_method"],
                "k_shot": ep["k_shot"],
                "seed": ep["seed"],
                "category_num": ep["category_num"],
                "n_test_images": ep["n_test_images"],
                "status": "ok" if ep["has_stats"] else "MISSING_STATS",
            }
            stats = ep["stats"] or {}
            for iou in IOU_TYPES:
                block = stats.get(iou, {}) if stats else {}
                for k in CSV_STAT_KEYS:
                    row[f"{iou}_{k}"] = block.get(k, "")
            timing = ep["timing"] or {}
            for k in TIMING_KEYS:
                row[k] = timing.get(k, "")
            writer.writerow(row)


def aggregate_by_kshot(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for ep in episodes:
        if not ep["has_stats"]:
            continue
        key = (ep["source_method"], ep["k_shot"])
        groups.setdefault(key, []).append(ep)

    out: Dict[str, Any] = {"groups": []}
    for (method, k), eps in sorted(
        groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] if kv[0][1] is not None else -1)
    ):
        entry: Dict[str, Any] = {
            "source_method": method,
            "k_shot": k,
            "n_episodes": len(eps),
            "seeds": sorted(e["seed"] for e in eps if e["seed"] is not None),
            "metrics": {},
            "timing": {},
        }
        for iou in IOU_TYPES:
            entry["metrics"][iou] = {}
            for stat in CSV_STAT_KEYS:
                vals = [(e["stats"].get(iou, {}) or {}).get(stat) for e in eps]
                mean, std = _mean_std([v for v in vals if v is not None])
                entry["metrics"][iou][stat] = {
                    "mean": mean,
                    "std": std,
                    "n": len([v for v in vals if v is not None]),
                }
        for tkey in TIMING_KEYS:
            vals = [(e["timing"] or {}).get(tkey) for e in eps]
            mean, std = _mean_std([v for v in vals if v is not None])
            entry["timing"][tkey] = {
                "mean": mean,
                "std": std,
                "n": len([v for v in vals if v is not None]),
            }
        out["groups"].append(entry)
    return out


def write_summary_csv(aggregate: Dict[str, Any], path: Path) -> None:
    fieldnames = ["source_method", "k_shot", "n_episodes"]
    for iou in IOU_TYPES:
        for stat in CSV_STAT_KEYS:
            fieldnames.extend([f"{iou}_{stat}_mean", f"{iou}_{stat}_std"])
    for tkey in TIMING_KEYS:
        fieldnames.extend([f"{tkey}_mean", f"{tkey}_std"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for g in aggregate["groups"]:
            row: Dict[str, Any] = {
                "source_method": g["source_method"],
                "k_shot": g["k_shot"],
                "n_episodes": g["n_episodes"],
            }
            for iou in IOU_TYPES:
                for stat in CSV_STAT_KEYS:
                    cell = g["metrics"][iou][stat]
                    row[f"{iou}_{stat}_mean"] = cell["mean"]
                    row[f"{iou}_{stat}_std"] = cell["std"]
            for tkey in TIMING_KEYS:
                cell = g["timing"][tkey]
                row[f"{tkey}_mean"] = cell["mean"]
                row[f"{tkey}_std"] = cell["std"]
            writer.writerow(row)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--packages-dir", type=Path, required=True, help="Parent dir containing episode package subdirs")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write collated CSV/JSON (default: <packages-dir>/fsodvfm_results)",
    )
    p.add_argument(
        "--work-subdir",
        type=str,
        default="fsodvfm_work",
        help="Per-package work directory name containing coco_eval_stats_.txt (default: fsodvfm_work)",
    )
    p.add_argument("--k-shots", type=str, default=None, help="Comma-separated k filter, e.g. '1,3,5,10'")
    p.add_argument("--seeds", type=str, default=None, help="Comma-separated seed filter, e.g. '10042,10043,10044'")
    return p.parse_args(argv)


def _csv_ints(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    packages_dir = args.packages_dir.expanduser().resolve()
    out_dir = (args.out_dir or (packages_dir / "fsodvfm_results")).expanduser().resolve()

    episodes = discover_episodes(
        packages_dir,
        work_subdir=args.work_subdir,
        k_shots=_csv_ints(args.k_shots),
        seeds=_csv_ints(args.seeds),
    )
    if not episodes:
        raise SystemExit(f"No episode packages with provenance.json found under {packages_dir}")

    aggregate = aggregate_by_kshot(episodes)

    write_episodes_csv(episodes, out_dir / "episodes.csv")
    write_summary_csv(aggregate, out_dir / "summary_by_kshot.csv")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "packages_dir": str(packages_dir),
                "work_subdir": args.work_subdir,
                "aggregate": aggregate,
            },
            f,
            indent=2,
        )

    n_ok = sum(1 for e in episodes if e["has_stats"])
    n_missing = len(episodes) - n_ok
    print(f"Collated {len(episodes)} episode(s): {n_ok} with metrics, {n_missing} missing.")
    if n_missing:
        for e in episodes:
            if not e["has_stats"]:
                print(f"  MISSING metrics: {e['package']} (k={e['k_shot']} seed={e['seed']})")
    print(f"Wrote:\n  {out_dir / 'episodes.csv'}\n  {out_dir / 'summary_by_kshot.csv'}\n  {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
