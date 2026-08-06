#!/usr/bin/env python3
"""Run FSOD-VFM on a single exported episode package.

Consumes an ExemplarSegmentation ``fsod_eval_nttt_episode_package_v1`` package
(same support / test / images used by SAM3 and no-time-to-train), converts the
COCO support JSON into FSOD-VFM's custom support format, invokes ``main.py``
(DINOv2 backbone), and writes NTTT-compatible metrics under
``<package>/fsodvfm_work/``.

Artifacts written to ``fsodvfm_work/``:
  * ``support_fsodvfm.json``  - converted support for main.py
  * ``predictions.json``      - COCO-format bbox detections
  * ``coco_eval_stats_.txt``  - bbox AP/AR (NTTT-parseable)
  * ``timing.json``           - wall-clock timing
  * ``run_invocation.json``   - audit record (argv, paths, hashes)

Example::

    python scripts/run_episodic_coco_package.py \\
      --package-dir /path/to/episode_packages/13393/k5_seed10042 \\
      --devices 0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_SUBDIR = "fsodvfm_work"

# COCOeval.stats index -> label used by NTTT collate STAT_NAME_TO_KEY
COCO_STAT_NAMES = [
    "AP IoU=0.50:0.95",
    "AP IoU=0.50",
    "AP IoU=0.75",
    "AP small",
    "AP medium",
    "AP large",
    "AR maxDets=1",
    "AR maxDets=10",
    "AR maxDets=100",
    "AR small",
    "AR medium",
    "AR large",
]

REQUIRED_CHECKPOINTS = [
    "checkpoints/sam2.1_hiera_large.pt",
    "checkpoints/dinov2_vitl14_pretrain.pth",
    "checkpoints/upn_large.pth",
]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def check_prerequisites(repo_root: Path) -> None:
    missing: List[str] = []
    if not (repo_root / "dinov2").is_dir():
        missing.append(f"dinov2/ (clone facebookresearch/dinov2 into {repo_root})")
    if not (repo_root / "main.py").is_file():
        missing.append("main.py")
    for rel in REQUIRED_CHECKPOINTS:
        if not (repo_root / rel).is_file():
            missing.append(rel)
    if missing:
        raise SystemExit(
            "FSOD-VFM prerequisites missing (run from repo root inside FSODVFM conda env):\n  - "
            + "\n  - ".join(missing)
        )


def resolve_images_dir(
    package_dir: Path,
    provenance: Dict[str, Any],
    *,
    override: Optional[Path] = None,
) -> Path:
    """Resolve the COCO images directory for an episode package.

    Priority:
      1. ``--images-dir`` override (for packages copied across machines)
      2. Package ``images/`` symlink (or directory) if it exists and resolves
      3. ``provenance.images_dir`` absolute path if it still exists
    """
    if override is not None:
        images_dir = override.expanduser().resolve()
        if not images_dir.is_dir():
            raise SystemExit(f"--images-dir does not exist or is not a directory: {images_dir}")
        return images_dir

    files = provenance.get("files") or {}
    symlink_name = files.get("images_symlink", "images")
    images_dir = package_dir / symlink_name
    if images_dir.exists():
        return images_dir.resolve()
    fallback = provenance.get("images_dir")
    if fallback and Path(fallback).exists():
        return Path(fallback).resolve()
    raise SystemExit(
        f"Could not resolve images dir: tried {images_dir} and "
        f"provenance.images_dir={fallback!r}. "
        f"Pass --images-dir /path/to/coco_13393/images when packages were copied from another machine."
    )


def resolve_support_path(package_dir: Path, provenance: Dict[str, Any]) -> Path:
    files = provenance.get("files") or {}
    # Prefer plain support.json (bboxes only needed); fall back to support_for_nttt / support_with_segm.
    candidates = [
        files.get("support", "support.json"),
        files.get("support_for_nttt"),
        "support_with_segm.json",
        "support.json",
    ]
    for name in candidates:
        if not name:
            continue
        path = package_dir / name
        if path.is_file():
            return path
    raise SystemExit(f"No support JSON found under {package_dir}")


def resolve_test_gt_path(package_dir: Path, provenance: Dict[str, Any]) -> Path:
    files = provenance.get("files") or {}
    name = files.get("test_gt", "test_gt.json")
    path = package_dir / name
    if not path.is_file():
        raise SystemExit(f"Missing test GT: {path}")
    return path


def category_names(provenance: Dict[str, Any]) -> List[str]:
    """Category names from provenance.categories (robust to spaces in names)."""
    cats = provenance.get("categories") or []
    names = [c["name"] for c in cats if isinstance(c, dict) and "name" in c]
    if names:
        return names
    # Fallback: only safe when names have no spaces / commas.
    csv = provenance.get("cat_names_csv") or ""
    if "," in csv:
        return [x.strip() for x in csv.split(",") if x.strip()]
    if csv.strip():
        return [csv.strip()]
    raise SystemExit("No categories found in provenance.json")


def convert_coco_support_to_fsodvfm(support_coco: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Convert COCO support JSON -> FSOD-VFM custom support format.

    Output: {class_name: [{"image": <file_name>, "bbox": [x,y,w,h]}, ...]}
    Image paths are relative to the images directory (used as --data_dir).
    """
    id_to_name = {c["id"]: c["name"] for c in support_coco.get("categories", [])}
    id_to_file = {img["id"]: img["file_name"] for img in support_coco.get("images", [])}
    out: Dict[str, List[Dict[str, Any]]] = {name: [] for name in id_to_name.values()}

    for ann in support_coco.get("annotations", []):
        cat_id = ann["category_id"]
        name = id_to_name.get(cat_id)
        if name is None:
            raise SystemExit(f"Annotation {ann.get('id')} has unknown category_id={cat_id}")
        img_id = ann["image_id"]
        file_name = id_to_file.get(img_id)
        if file_name is None:
            raise SystemExit(f"Annotation {ann.get('id')} references missing image_id={img_id}")
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            raise SystemExit(f"Annotation {ann.get('id')} missing valid bbox")
        out[name].append({"image": file_name, "bbox": list(bbox)})

    # Drop empty classes (shouldn't happen for valid packages).
    return {k: v for k, v in out.items() if v}


def validate_support_counts(
    support_fsod: Dict[str, List[Dict[str, Any]]],
    provenance: Dict[str, Any],
) -> None:
    k = provenance.get("k_shot")
    if k is None:
        return
    for name, shots in support_fsod.items():
        if len(shots) != k:
            raise SystemExit(
                f"Support count mismatch for class {name!r}: got {len(shots)} shots, expected k_shot={k}"
            )


def write_coco_eval_stats_txt(stats_path: Path, bbox_stats: Sequence[float]) -> None:
    """Write NTTT-parseable coco_eval_stats_.txt (bbox section only)."""
    lines = ["===== BBOX RESULTS ====="]
    for name, val in zip(COCO_STAT_NAMES, bbox_stats):
        lines.append(f"{name}: {float(val):.4f}")
    lines.append("")
    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_bbox(
    gt_json_path: Path,
    pred_json_path: Path,
    target_categories: List[str],
    stats_out: Path,
) -> List[float]:
    """Run pycocotools bbox COCOeval and write coco_eval_stats_.txt."""
    import pycocotools.coco
    import pycocotools.cocoeval

    if not pred_json_path.is_file():
        raise SystemExit(f"Predictions file missing: {pred_json_path}")

    preds = _load_json(pred_json_path)
    if not preds:
        # Empty predictions: write zero stats so collation still works.
        zeros = [0.0] * len(COCO_STAT_NAMES)
        write_coco_eval_stats_txt(stats_out, zeros)
        return zeros

    coco_gt = pycocotools.coco.COCO(str(gt_json_path))
    if "info" not in coco_gt.dataset:
        coco_gt.dataset["info"] = {"description": "Auto-added info"}
    if "licenses" not in coco_gt.dataset:
        coco_gt.dataset["licenses"] = []

    coco_dt = coco_gt.loadRes(preds)
    coco_eval = pycocotools.cocoeval.COCOeval(coco_gt, coco_dt, iouType="bbox")
    if target_categories:
        cat_ids = coco_gt.getCatIds(catNms=target_categories)
        if not cat_ids:
            raise SystemExit(
                f"No category ids found for target_categories={target_categories!r} in {gt_json_path}"
            )
        coco_eval.params.catIds = cat_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = [float(x) for x in coco_eval.stats.tolist()]
    write_coco_eval_stats_txt(stats_out, stats)
    return stats


def build_main_cmd(
    *,
    support_json: Path,
    images_dir: Path,
    test_json: Path,
    pred_json: Path,
    target_categories: List[str],
    min_threshold: float,
    diffusion_steps: int,
    alp: float,
    lamb: float,
    model_version: str,
    dinov2_checkpoint_dir: Path,
    repo_or_dir: Path,
) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "main.py"),
        "--json_path",
        str(support_json),
        "--data_dir",
        str(images_dir),
        "--test_json",
        str(test_json),
        "--test_img_dir",
        str(images_dir),
        "--pred_json",
        str(pred_json),
        "--feat_extractor_name",
        "DINOV2",
        "--model_version",
        model_version,
        "--repo_or_dir",
        str(repo_or_dir),
        "--dinov2_checkpoint_dir",
        str(dinov2_checkpoint_dir),
        "--min_threshold",
        str(min_threshold),
        "--diffusion_steps",
        str(diffusion_steps),
        "--alp",
        str(alp),
        "--lamb",
        str(lamb),
        "--filter_by_categories",
        "--target_categories",
        *target_categories,
    ]
    return cmd


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--package-dir", type=Path, required=True, help="Episode package directory")
    p.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help=(
            "Override path to COCO images (shared across packages). "
            "Use when package images/ symlinks or provenance.images_dir point at another machine."
        ),
    )
    p.add_argument("--devices", type=str, default="0", help="CUDA_VISIBLE_DEVICES value (e.g. '0' or '0,1')")
    p.add_argument("--force", action="store_true", help="Re-run even if coco_eval_stats_.txt exists")
    p.add_argument("--dry-run", action="store_true", help="Convert support + print command without running main.py")
    p.add_argument("--min-threshold", type=float, default=0.01)
    p.add_argument("--diffusion-steps", type=int, default=30)
    p.add_argument("--alp", type=float, default=0.3)
    p.add_argument("--lamb", type=float, default=0.5)
    p.add_argument("--model-version", type=str, default="dinov2_vitl14")
    p.add_argument(
        "--dinov2-checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "checkpoints",
        help="Directory containing dinov2_vitl14_pretrain.pth",
    )
    p.add_argument(
        "--repo-or-dir",
        type=Path,
        default=REPO_ROOT / "dinov2",
        help="DINOv2 code directory for torch.hub.load",
    )
    p.add_argument(
        "--skip-prereq-check",
        action="store_true",
        help="Skip checkpoint / dinov2 directory checks (useful for dry-run)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    package_dir = args.package_dir.expanduser().resolve()
    if not package_dir.is_dir():
        raise SystemExit(f"Package dir does not exist: {package_dir}")

    prov_path = package_dir / "provenance.json"
    if not prov_path.is_file():
        raise SystemExit(f"Missing provenance.json in {package_dir}")
    provenance = _load_json(prov_path)

    work_dir = package_dir / WORK_SUBDIR
    stats_file = work_dir / "coco_eval_stats_.txt"
    if stats_file.is_file() and not args.force and not args.dry_run:
        print(f"SKIP (already done): {package_dir.name} -> {stats_file}")
        return

    if not args.skip_prereq_check and not args.dry_run:
        check_prerequisites(REPO_ROOT)

    images_dir = resolve_images_dir(package_dir, provenance, override=args.images_dir)
    support_coco_path = resolve_support_path(package_dir, provenance)
    test_gt_path = resolve_test_gt_path(package_dir, provenance)
    cat_names = category_names(provenance)

    work_dir.mkdir(parents=True, exist_ok=True)
    support_coco = _load_json(support_coco_path)
    support_fsod = convert_coco_support_to_fsodvfm(support_coco)
    validate_support_counts(support_fsod, provenance)

    support_fsod_path = work_dir / "support_fsodvfm.json"
    _dump_json(support_fsod_path, support_fsod)
    print(f"Wrote converted support: {support_fsod_path}")
    for name, shots in support_fsod.items():
        print(f"  class={name!r} n_shots={len(shots)}")

    pred_json = work_dir / "predictions.json"
    cmd = build_main_cmd(
        support_json=support_fsod_path,
        images_dir=images_dir,
        test_json=test_gt_path,
        pred_json=pred_json,
        target_categories=cat_names,
        min_threshold=args.min_threshold,
        diffusion_steps=args.diffusion_steps,
        alp=args.alp,
        lamb=args.lamb,
        model_version=args.model_version,
        dinov2_checkpoint_dir=args.dinov2_checkpoint_dir.expanduser().resolve(),
        repo_or_dir=args.repo_or_dir.expanduser().resolve(),
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.devices

    invocation = {
        "package_dir": str(package_dir),
        "k_shot": provenance.get("k_shot"),
        "seed": provenance.get("seed"),
        "categories": provenance.get("categories"),
        "cat_names": cat_names,
        "images_dir": str(images_dir),
        "support_coco": str(support_coco_path),
        "support_fsodvfm": str(support_fsod_path),
        "test_gt": str(test_gt_path),
        "pred_json": str(pred_json),
        "cmd": cmd,
        "cuda_visible_devices": args.devices,
        "hashes": provenance.get("hashes"),
        "hyperparams": {
            "min_threshold": args.min_threshold,
            "diffusion_steps": args.diffusion_steps,
            "alp": args.alp,
            "lamb": args.lamb,
            "model_version": args.model_version,
            "feat_extractor_name": "DINOV2",
        },
    }
    _dump_json(work_dir / "run_invocation.json", invocation)

    if args.dry_run:
        print("DRY-RUN command:")
        print("  CUDA_VISIBLE_DEVICES=" + args.devices)
        print("  " + " ".join(repr(c) if " " in str(c) else str(c) for c in cmd))
        return

    n_test_images = provenance.get("n_test_images")
    print(f"Running FSOD-VFM on {package_dir.name} (k={provenance.get('k_shot')} seed={provenance.get('seed')})")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    total_s = time.perf_counter() - t0
    invocation["main_returncode"] = proc.returncode
    _dump_json(work_dir / "run_invocation.json", invocation)

    if proc.returncode != 0:
        if pred_json.is_file() and _load_json(pred_json):
            print(
                f"WARNING: main.py exited {proc.returncode} (often segm eval on GT with "
                "missing/invalid segmentation). predictions.json exists; continuing with "
                "bbox-only episodic eval."
            )
        else:
            raise SystemExit(f"main.py failed with returncode={proc.returncode} and no predictions.json")

    # Independent bbox eval -> NTTT-compatible stats file (decoupled from main.py's ./results/).
    evaluate_bbox(test_gt_path, pred_json, cat_names, stats_file)

    timing = {
        "fill_memory_s": None,
        "postprocess_memory_s": None,
        "test_s": round(total_s, 3),
        "total_s": round(total_s, 3),
        "test_s_per_image": round(total_s / n_test_images, 4) if n_test_images else None,
        "n_test_images": n_test_images,
        "timing_note": (
            "Wall-clock for the full main.py process (model load + support prototypes + query "
            "inference + internal COCO eval). fill_memory/postprocess are N/A for FSOD-VFM "
            "(single-process pipeline)."
        ),
    }
    _dump_json(work_dir / "timing.json", timing)
    print(f"Done: stats={stats_file} total_s={total_s:.1f}")


if __name__ == "__main__":
    main()
