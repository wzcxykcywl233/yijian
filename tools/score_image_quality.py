"""Score source and augmented character images without training a classifier.

The scores are intended for quality control before/after background fusion:

- source images: filter unreadable or badly segmented single-character crops
- generated images: flag failed fusion samples such as filled dark rectangles

Metrics include PSNR, SSIM, entropy, blur/sharpness, contrast, edge density,
dark/white pixel ratios, and foreground-mask shape statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from character_region_extractor import ExtractConfig, read_image
from generate_character_samples import make_glyph_alpha, normalize_glyph_darkness


DEFAULT_MANIFEST_FIELDS = [
    "path",
    "char",
    "char_code",
    "source_label",
    "source_image",
    "background_source",
    "background",
    "method",
    "pre_extract_enhance",
    "angle",
    "scale",
]


@dataclass(frozen=True)
class Thresholds:
    min_source_score: float
    min_generated_score: float
    min_alpha_ratio: float
    max_alpha_ratio: float
    min_darkness_mean: float
    max_bbox_fill_ratio: float
    max_dark_ratio: float
    max_white_ratio: float
    min_contrast_std: float
    min_laplacian_var: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def normalize_path(path: Path) -> str:
    try:
        return str(path.resolve()).casefold()
    except OSError:
        return str(path).casefold()


def entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    prob = hist / max(float(hist.sum()), 1.0)
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    err = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if err <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(err)))


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab
    score = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2))
    return float(np.clip(np.mean(score), -1.0, 1.0))


def bounded(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def closeness(value: float, target: float, tolerance: float) -> float:
    return float(np.clip(1.0 - abs(value - target) / max(tolerance, 1e-6), 0.0, 1.0))


def mask_stats(image: np.ndarray, cfg: ExtractConfig) -> dict[str, float | bool]:
    alpha = make_glyph_alpha(image, cfg)
    darkness = normalize_glyph_darkness(image, alpha)
    glyph = alpha > 16
    alpha_ratio = float(np.count_nonzero(glyph) / glyph.size)
    if not np.any(glyph):
        return {
            "alpha_ratio": alpha_ratio,
            "darkness_mean": 0.0,
            "bbox_fill_ratio": 1.0,
            "bbox_area_ratio": 1.0,
            "component_count": 0,
            "mask_ok": False,
        }

    component_count, _, stats, _ = cv2.connectedComponentsWithStats(glyph.astype(np.uint8), 8)
    if component_count <= 1:
        bbox_fill_ratio = 1.0
        bbox_area_ratio = 1.0
    else:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = float(stats[largest, cv2.CC_STAT_AREA])
        width = float(stats[largest, cv2.CC_STAT_WIDTH])
        height = float(stats[largest, cv2.CC_STAT_HEIGHT])
        bbox_area = max(width * height, 1.0)
        bbox_fill_ratio = area / bbox_area
        bbox_area_ratio = bbox_area / float(glyph.size)

    return {
        "alpha_ratio": alpha_ratio,
        "darkness_mean": float(np.mean(darkness[glyph])),
        "bbox_fill_ratio": float(bbox_fill_ratio),
        "bbox_area_ratio": float(bbox_area_ratio),
        "component_count": int(component_count - 1),
        "mask_ok": True,
    }


def image_metrics(image: np.ndarray, thresholds: Thresholds, cfg: ExtractConfig) -> dict[str, object]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ent = entropy(gray)
    contrast = float(np.std(gray))
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges) / edges.size)
    dark_ratio = float(np.count_nonzero(gray < 55) / gray.size)
    white_ratio = float(np.count_nonzero(gray > 245) / gray.size)
    stats = mask_stats(image, cfg)

    alpha_ratio = float(stats["alpha_ratio"])
    darkness_mean = float(stats["darkness_mean"])
    bbox_fill_ratio = float(stats["bbox_fill_ratio"])
    bbox_area_ratio = float(stats["bbox_area_ratio"])

    alpha_score = 1.0 if thresholds.min_alpha_ratio <= alpha_ratio <= thresholds.max_alpha_ratio else 0.0
    darkness_score = bounded(darkness_mean, thresholds.min_darkness_mean, thresholds.min_darkness_mean * 8)
    rectangle_score = 0.0 if bbox_fill_ratio > thresholds.max_bbox_fill_ratio and bbox_area_ratio > 0.08 else 1.0
    contrast_score = bounded(contrast, thresholds.min_contrast_std, 64.0)
    sharpness_score = bounded(math.log1p(lap), math.log1p(thresholds.min_laplacian_var), math.log1p(1200.0))
    edge_score = 1.0 if 0.003 <= edge_density <= 0.35 else closeness(edge_density, 0.08, 0.08)
    entropy_score = bounded(ent, 3.0, 8.0)
    dark_score = 1.0 - bounded(dark_ratio, thresholds.max_dark_ratio, min(1.0, thresholds.max_dark_ratio + 0.35))
    white_score = 1.0 - bounded(white_ratio, thresholds.max_white_ratio, min(1.0, thresholds.max_white_ratio + 0.35))

    no_ref_score = (
        0.16 * entropy_score
        + 0.16 * contrast_score
        + 0.16 * sharpness_score
        + 0.12 * edge_score
        + 0.18 * alpha_score
        + 0.12 * darkness_score
        + 0.07 * rectangle_score
        + 0.03 * min(dark_score, white_score)
    )

    return {
        "entropy": round(ent, 6),
        "contrast_std": round(contrast, 6),
        "laplacian_var": round(lap, 6),
        "edge_density": round(edge_density, 6),
        "dark_ratio": round(dark_ratio, 6),
        "white_ratio": round(white_ratio, 6),
        "alpha_ratio": round(alpha_ratio, 6),
        "darkness_mean": round(darkness_mean, 6),
        "bbox_fill_ratio": round(bbox_fill_ratio, 6),
        "bbox_area_ratio": round(bbox_area_ratio, 6),
        "component_count": stats["component_count"],
        "no_ref_score": round(no_ref_score, 6),
        "alpha_ok": alpha_score > 0,
        "rectangle_ok": rectangle_score > 0,
        "contrast_ok": contrast >= thresholds.min_contrast_std,
        "sharpness_ok": lap >= thresholds.min_laplacian_var,
        "darkness_ok": darkness_mean >= thresholds.min_darkness_mean,
        "dark_ratio_ok": dark_ratio <= thresholds.max_dark_ratio,
        "white_ratio_ok": white_ratio <= thresholds.max_white_ratio,
    }


def compare_to_reference(image: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    if ref_gray.shape != gray.shape:
        ref_gray = cv2.resize(ref_gray, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
    return {
        "psnr": round(psnr(gray, ref_gray), 6),
        "ssim": round(ssim_gray(gray, ref_gray), 6),
        "entropy_delta": round(entropy(gray) - entropy(ref_gray), 6),
    }


def quality_reasons(metrics: dict[str, object], source_kind: str) -> str:
    reasons = []
    for key, reason in [
        ("alpha_ok", "bad_alpha_area"),
        ("rectangle_ok", "filled_rectangle_mask"),
        ("contrast_ok", "low_contrast"),
        ("sharpness_ok", "blurred"),
        ("darkness_ok", "weak_ink"),
        ("dark_ratio_ok", "too_dark"),
        ("white_ratio_ok", "too_white"),
    ]:
        if metrics.get(key) is False:
            reasons.append(reason)
    if source_kind == "generated" and metrics.get("ssim") != "":
        entropy_delta = abs(float(metrics.get("entropy_delta", 0.0)))
        if entropy_delta > 1.8:
            reasons.append("entropy_shift_too_large")
    return "+".join(reasons)


def final_score(metrics: dict[str, object], kind: str) -> float:
    no_ref = float(metrics["no_ref_score"])
    if kind == "source" or metrics.get("psnr") == "":
        return no_ref
    psnr_score = bounded(float(metrics["psnr"]), 12.0, 32.0)
    ssim_score = bounded(float(metrics["ssim"]), 0.45, 0.98)
    entropy_score = closeness(float(metrics["entropy_delta"]), 0.15, 1.2)
    return round(0.55 * no_ref + 0.15 * psnr_score + 0.20 * ssim_score + 0.10 * entropy_score, 6)


def source_image_path(data_root: Path, row: dict[str, str]) -> Path:
    return data_root / row.get("image_path", "")


def manifest_image_path(manifest: Path, row: dict[str, str]) -> Path:
    return manifest.parent / row.get("path", "")


def manifest_reference_path(row: dict[str, str]) -> Path | None:
    value = row.get("source_image", "")
    if not value:
        return None
    return Path(value)


def score_image(path: Path, thresholds: Thresholds, cfg: ExtractConfig, reference: Path | None, kind: str) -> dict[str, object]:
    image = read_image(path, cv2.IMREAD_COLOR)
    if image is None:
        return {"readable": False, "quality_score": 0.0, "quality_ok": False, "quality_reasons": "unreadable"}

    metrics = image_metrics(image, thresholds, cfg)
    if reference is not None and reference.exists():
        ref_image = read_image(reference, cv2.IMREAD_COLOR)
        if ref_image is not None:
            metrics.update(compare_to_reference(image, ref_image))
        else:
            metrics.update({"psnr": "", "ssim": "", "entropy_delta": ""})
    else:
        metrics.update({"psnr": "", "ssim": "", "entropy_delta": ""})

    score = final_score(metrics, kind)
    threshold = thresholds.min_source_score if kind == "source" else thresholds.min_generated_score
    reasons = quality_reasons(metrics, kind)
    hard_ok = not reasons
    return {
        "readable": True,
        **metrics,
        "quality_score": round(score, 6),
        "quality_ok": bool(score >= threshold and hard_ok),
        "quality_reasons": reasons,
    }


def discover_manifests(experiment_dir: Path | None, manifests: list[Path]) -> list[Path]:
    found = list(manifests)
    if experiment_dir is not None:
        found.extend(sorted((experiment_dir / "augment").glob("*/generated_samples.csv")))
    unique = []
    seen = set()
    for path in found:
        key = normalize_path(path)
        if path.exists() and key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def score_sources(data_root: Path, clean_samples: Path, thresholds: Thresholds, cfg: ExtractConfig, limit: int | None) -> tuple[list[dict[str, object]], dict[str, bool]]:
    rows = read_csv(clean_samples)
    if limit is not None:
        rows = rows[:limit]
    scored = []
    ok_by_path: dict[str, bool] = {}
    for row in rows:
        path = source_image_path(data_root, row)
        metrics = score_image(path, thresholds, cfg, None, "source")
        record = {
            "kind": "source",
            "char": row.get("char", ""),
            "source_label": row.get("source_label", ""),
            "image_path": row.get("image_path", ""),
            "abs_path": str(path),
            **metrics,
        }
        scored.append(record)
        ok_by_path[normalize_path(path)] = bool(metrics["quality_ok"])
    return scored, ok_by_path


def score_generated(
    manifests: list[Path],
    thresholds: Thresholds,
    cfg: ExtractConfig,
    source_ok_by_path: dict[str, bool],
    require_source_ok: bool,
    limit: int | None,
) -> tuple[list[dict[str, object]], dict[Path, list[dict[str, str]]]]:
    scored = []
    filtered_by_manifest: dict[Path, list[dict[str, str]]] = {}
    for manifest in manifests:
        rows = read_csv(manifest)
        if limit is not None:
            rows = rows[:limit]
        kept_rows: list[dict[str, str]] = []
        for row in rows:
            path = manifest_image_path(manifest, row)
            ref = manifest_reference_path(row)
            metrics = score_image(path, thresholds, cfg, ref, "generated")
            source_ok = True
            if ref is not None:
                key = normalize_path(ref)
                source_ok = source_ok_by_path.get(key, True)
            generated_ok = bool(metrics["quality_ok"]) and (source_ok or not require_source_ok)
            if generated_ok:
                kept_rows.append(row)
            record = {
                "kind": "generated",
                "char": row.get("char", ""),
                "source_label": row.get("source_label", ""),
                "image_path": row.get("path", ""),
                "abs_path": str(path),
                "manifest": str(manifest),
                "method": row.get("pre_extract_enhance", "") or row.get("method", ""),
                "background_source": row.get("background_source", ""),
                "reference_path": str(ref) if ref is not None else "",
                "source_quality_ok": source_ok,
                **metrics,
                "quality_ok": generated_ok,
                "quality_reasons": metrics["quality_reasons"] + ("+bad_source_image" if require_source_ok and not source_ok else ""),
            }
            scored.append(record)
        filtered_by_manifest[manifest] = kept_rows
    return scored, filtered_by_manifest


def write_filtered_clean_samples(clean_samples: Path, data_root: Path, out_csv: Path, source_ok_by_path: dict[str, bool]) -> int:
    rows = []
    for row in read_csv(clean_samples):
        if source_ok_by_path.get(normalize_path(source_image_path(data_root, row)), False):
            rows.append(row)
    fieldnames = list(rows[0].keys()) if rows else list(read_csv(clean_samples)[0].keys())
    write_csv(out_csv, rows, fieldnames)
    return len(rows)


def write_filtered_manifests(out_dir: Path, filtered: dict[Path, list[dict[str, str]]]) -> dict[str, int]:
    counts = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for manifest, rows in filtered.items():
        name = manifest.parent.name
        out_path = out_dir / f"{name}_filtered_generated_samples.csv"
        fieldnames = list(rows[0].keys()) if rows else DEFAULT_MANIFEST_FIELDS
        write_csv(out_path, rows, fieldnames)
        counts[str(out_path)] = len(rows)
    return counts


def summarize_quality(rows: list[dict[str, object]], group_key: str) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(group_key, "") or "unknown"), []).append(row)

    summaries = []
    for group, items in sorted(grouped.items()):
        total = len(items)
        ok = sum(1 for row in items if row.get("quality_ok") is True)
        avg_score = sum(float(row.get("quality_score", 0.0) or 0.0) for row in items) / max(total, 1)
        avg_entropy = sum(float(row.get("entropy", 0.0) or 0.0) for row in items) / max(total, 1)
        avg_ssim_values = [float(row["ssim"]) for row in items if row.get("ssim") not in ("", None)]
        avg_psnr_values = [float(row["psnr"]) for row in items if row.get("psnr") not in ("", None)]
        summaries.append(
            {
                group_key: group,
                "total": total,
                "quality_ok": ok,
                "quality_ok_rate": round(ok / total, 6) if total else 0.0,
                "avg_quality_score": round(avg_score, 6),
                "avg_entropy": round(avg_entropy, 6),
                "avg_ssim": round(sum(avg_ssim_values) / len(avg_ssim_values), 6) if avg_ssim_values else "",
                "avg_psnr": round(sum(avg_psnr_values) / len(avg_psnr_values), 6) if avg_psnr_values else "",
            }
        )
    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score source and generated character image quality.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--clean_samples", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--experiment_dir", type=Path, help="Existing experiment directory containing augment/*/generated_samples.csv.")
    parser.add_argument("--manifest", action="append", type=Path, default=[], help="Additional generated_samples.csv to score.")
    parser.add_argument("--filtered_clean_samples", type=Path, help="Write clean_samples filtered by source-image quality.")
    parser.add_argument("--filtered_manifest_dir", type=Path, help="Write filtered generated manifests under this directory.")
    parser.add_argument("--require_source_ok_for_manifest", action="store_true", help="Drop generated samples whose source image failed source quality checks.")
    parser.add_argument("--limit", type=int, help="Debug limit per input CSV.")
    parser.add_argument("--min_source_score", type=float, default=0.38)
    parser.add_argument("--min_generated_score", type=float, default=0.36)
    parser.add_argument("--min_alpha_ratio", type=float, default=0.003)
    parser.add_argument("--max_alpha_ratio", type=float, default=0.45)
    parser.add_argument("--min_darkness_mean", type=float, default=0.015)
    parser.add_argument("--max_bbox_fill_ratio", type=float, default=0.72)
    parser.add_argument("--max_dark_ratio", type=float, default=0.78)
    parser.add_argument("--max_white_ratio", type=float, default=0.92)
    parser.add_argument("--min_contrast_std", type=float, default=7.0)
    parser.add_argument("--min_laplacian_var", type=float, default=3.0)
    parser.add_argument("--mask_feather", type=int, default=1)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    thresholds = Thresholds(
        min_source_score=args.min_source_score,
        min_generated_score=args.min_generated_score,
        min_alpha_ratio=args.min_alpha_ratio,
        max_alpha_ratio=args.max_alpha_ratio,
        min_darkness_mean=args.min_darkness_mean,
        max_bbox_fill_ratio=args.max_bbox_fill_ratio,
        max_dark_ratio=args.max_dark_ratio,
        max_white_ratio=args.max_white_ratio,
        min_contrast_std=args.min_contrast_std,
        min_laplacian_var=args.min_laplacian_var,
    )
    cfg = ExtractConfig(backend="opencv", feather=args.mask_feather, morph_kernel=3, grabcut_iters=2)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_rows, source_ok_by_path = score_sources(args.data_root, args.clean_samples, thresholds, cfg, args.limit)
    source_fields = list(source_rows[0].keys()) if source_rows else ["kind", "char", "image_path", "quality_score", "quality_ok"]
    write_csv(args.out_dir / "source_image_quality.csv", source_rows, source_fields)
    source_summary = summarize_quality(source_rows, "source_label")
    write_csv(args.out_dir / "source_quality_summary.csv", source_summary, list(source_summary[0].keys()) if source_summary else ["source_label", "total"])

    manifests = discover_manifests(args.experiment_dir, args.manifest)
    generated_rows, filtered = score_generated(manifests, thresholds, cfg, source_ok_by_path, args.require_source_ok_for_manifest, args.limit)
    generated_summary = []
    if generated_rows:
        write_csv(args.out_dir / "generated_image_quality.csv", generated_rows, list(generated_rows[0].keys()))
        generated_summary = summarize_quality(generated_rows, "method")
        write_csv(args.out_dir / "generated_quality_summary.csv", generated_summary, list(generated_summary[0].keys()))

    filtered_clean_count = ""
    if args.filtered_clean_samples is not None:
        filtered_clean_count = write_filtered_clean_samples(args.clean_samples, args.data_root, args.filtered_clean_samples, source_ok_by_path)

    filtered_manifest_counts = {}
    if args.filtered_manifest_dir is not None:
        filtered_manifest_counts = write_filtered_manifests(args.filtered_manifest_dir, filtered)

    summary = {
        "source_images": len(source_rows),
        "source_ok": sum(1 for row in source_rows if row.get("quality_ok") is True),
        "generated_images": len(generated_rows),
        "generated_ok": sum(1 for row in generated_rows if row.get("quality_ok") is True),
        "source_summary": source_summary,
        "generated_summary": generated_summary,
        "manifests": [str(path) for path in manifests],
        "filtered_clean_samples": filtered_clean_count,
        "filtered_manifests": filtered_manifest_counts,
        "out_dir": str(args.out_dir),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
