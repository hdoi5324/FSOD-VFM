#!/usr/bin/env python3
"""Run FSOD-VFM over every exported episode package, then collate.

Discovers episode packages (each a subdir of ``--packages-dir`` containing
``provenance.json``), runs ``scripts/run_episodic_coco_package.py`` for each,
continues past failures, and finally invokes
``scripts/collate_episodic_results.py`` to produce mean/std-by-k-shot metrics
and timing.

Example::

    python scripts/run_episodic_coco_batch.py \\
      --packages-dir /path/to/ExemplarSegmentation/episode_packages/13393 \\
      --devices 0 \\
      --k-shots 5,10 \\
      --seeds 10042,10043,10044

Notes:
  * Runs episodes sequentially on the given --devices.
  * Skips episodes that already have ``fsodvfm_work/coco_eval_stats_.txt``
    unless --force.
  * Must be run from the FSOD-VFM repo root (child uses relative checkpoints /
    dinov2). This wrapper enforces that by setting cwd to the repo root.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_episodic_coco_package.py"
COLLATE_SCRIPT = REPO_ROOT / "scripts" / "collate_episodic_results.py"
WORK_SUBDIR = "fsodvfm_work"


def _csv_ints(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def discover_packages(
    packages_dir: Path,
    *,
    k_shots: Optional[List[int]],
    seeds: Optional[List[int]],
) -> List[Dict[str, Any]]:
    pkgs: List[Dict[str, Any]] = []
    for prov_path in sorted(packages_dir.glob("*/provenance.json")):
        with prov_path.open("r", encoding="utf-8") as f:
            prov = json.load(f)
        k = prov.get("k_shot")
        seed = prov.get("seed")
        if k_shots is not None and k not in k_shots:
            continue
        if seeds is not None and seed not in seeds:
            continue
        pkgs.append(
            {
                "dir": prov_path.parent,
                "k_shot": k,
                "seed": seed,
                "cat_names_csv": prov.get("cat_names_csv", ""),
                "category_num": prov.get("category_num"),
            }
        )
    return pkgs


def already_done(pkg_dir: Path) -> bool:
    return (pkg_dir / WORK_SUBDIR / "coco_eval_stats_.txt").is_file()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--packages-dir", type=Path, required=True, help="Parent dir containing episode package subdirs")
    p.add_argument("--devices", type=str, default="0", help="CUDA devices passed through to the per-episode script")
    p.add_argument("--k-shots", type=str, default=None, help="Comma-separated k filter, e.g. '1,3,5,10'")
    p.add_argument("--seeds", type=str, default=None, help="Comma-separated seed filter, e.g. '10042,10043,10044'")
    p.add_argument("--force", action="store_true", help="Re-run episodes even if results already exist")
    p.add_argument("--dry-run", action="store_true", help="List what would run without executing")
    p.add_argument("--skip-collate", action="store_true", help="Do not run the collator at the end")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Collation output dir (default: <packages-dir>/fsodvfm_results)",
    )
    # Hyperparams forwarded to the package runner (match run_coco.sh defaults).
    p.add_argument("--min-threshold", type=float, default=0.01)
    p.add_argument("--diffusion-steps", type=int, default=30)
    p.add_argument("--alp", type=float, default=0.3)
    p.add_argument("--lamb", type=float, default=0.5)
    p.add_argument("--model-version", type=str, default="dinov2_vitl14")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    packages_dir = args.packages_dir.expanduser().resolve()
    out_dir = (args.out_dir or (packages_dir / "fsodvfm_results")).expanduser().resolve()

    if not RUN_SCRIPT.is_file():
        raise SystemExit(f"Missing package runner: {RUN_SCRIPT}")
    if not COLLATE_SCRIPT.is_file():
        raise SystemExit(f"Missing collate script: {COLLATE_SCRIPT}")

    pkgs = discover_packages(
        packages_dir,
        k_shots=_csv_ints(args.k_shots),
        seeds=_csv_ints(args.seeds),
    )
    if not pkgs:
        raise SystemExit(f"No episode packages with provenance.json found under {packages_dir}")

    print(f"Discovered {len(pkgs)} episode package(s) under {packages_dir}")
    results: List[Dict[str, Any]] = []
    for i, pkg in enumerate(pkgs, 1):
        pkg_dir = pkg["dir"]
        tag = f"{pkg_dir.name} (k={pkg['k_shot']} seed={pkg['seed']})"
        if not args.force and already_done(pkg_dir):
            print(f"[{i}/{len(pkgs)}] SKIP (already done): {tag}")
            results.append({"package": pkg_dir.name, "status": "skipped", "wall_s": None})
            continue

        cmd = [
            sys.executable,
            str(RUN_SCRIPT),
            "--package-dir",
            str(pkg_dir),
            "--devices",
            args.devices,
            "--min-threshold",
            str(args.min_threshold),
            "--diffusion-steps",
            str(args.diffusion_steps),
            "--alp",
            str(args.alp),
            "--lamb",
            str(args.lamb),
            "--model-version",
            args.model_version,
        ]
        if args.force:
            cmd.append("--force")
        if args.dry_run:
            cmd.append("--dry-run")
            cmd.append("--skip-prereq-check")

        if args.dry_run:
            print(f"[{i}/{len(pkgs)}] DRY-RUN: {tag}")
            # Still invoke package runner in dry-run mode so support conversion is exercised.
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
            status = "dry-run" if proc.returncode == 0 else f"FAILED(rc={proc.returncode})"
            results.append({"package": pkg_dir.name, "status": status, "wall_s": None, "returncode": proc.returncode})
            continue

        print(f"[{i}/{len(pkgs)}] RUN: {tag}")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        wall_s = round(time.perf_counter() - t0, 3)
        status = "ok" if proc.returncode == 0 else f"FAILED(rc={proc.returncode})"
        print(f"[{i}/{len(pkgs)}] {status} in {wall_s:.1f}s: {tag}")
        results.append(
            {
                "package": pkg_dir.name,
                "k_shot": pkg["k_shot"],
                "seed": pkg["seed"],
                "status": status,
                "wall_s": wall_s,
                "returncode": proc.returncode,
            }
        )

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = out_dir / "batch_manifest.json"
        with manifest.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "packages_dir": str(packages_dir),
                    "devices": args.devices,
                    "work_subdir": WORK_SUBDIR,
                    "episodes": results,
                },
                f,
                indent=2,
            )
        n_fail = sum(1 for r in results if str(r["status"]).startswith("FAILED"))
        n_ok = sum(1 for r in results if r["status"] == "ok")
        n_skip = sum(1 for r in results if r["status"] == "skipped")
        print(f"\nBatch done: {n_ok} ok, {n_skip} skipped, {n_fail} failed. Manifest: {manifest}")
        if n_fail:
            for r in results:
                if str(r["status"]).startswith("FAILED"):
                    print(f"  FAILED: {r['package']}")

    if args.skip_collate or args.dry_run:
        return

    print("\nCollating results ...")
    collate_cmd = [
        sys.executable,
        str(COLLATE_SCRIPT),
        "--packages-dir",
        str(packages_dir),
        "--out-dir",
        str(out_dir),
        "--work-subdir",
        WORK_SUBDIR,
    ]
    if args.k_shots:
        collate_cmd += ["--k-shots", args.k_shots]
    if args.seeds:
        collate_cmd += ["--seeds", args.seeds]
    subprocess.run(collate_cmd, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
