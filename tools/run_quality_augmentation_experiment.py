"""Run augmentation variants and evaluate them with image-quality scores only."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def generate_cmd(args: argparse.Namespace, clean_samples: Path, out_dir: Path, method: str) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "generate_character_samples.py"),
        "--data_root",
        str(args.data_root),
        "--clean_samples",
        str(clean_samples),
        "--background_root",
        str(args.background_root),
        "--out_dir",
        str(out_dir),
        "--target_count",
        str(args.target_count),
        "--image_size",
        str(args.image_size),
        "--pre_extract_enhance",
        method,
        "--background_source_policy",
        "cycle_all_sources",
        "--mask_feather",
        str(args.mask_feather),
        "--ink_strength_min",
        str(args.ink_strength_min),
        "--ink_strength_max",
        str(args.ink_strength_max),
        "--alpha_power",
        str(args.alpha_power),
        "--darkness_gamma",
        str(args.darkness_gamma),
        "--source_bg_strength",
        str(args.source_bg_strength),
        "--white_alpha_suppression",
        str(args.white_alpha_suppression),
        "--sample_attempts",
        str(args.sample_attempts),
        "--max_bbox_fill_ratio",
        str(args.max_bbox_fill_ratio),
        "--seed",
        str(args.seed),
    ]
    if args.rare_chars is not None:
        cmd.extend(["--rare_chars", str(args.rare_chars)])
    if args.limit_chars is not None:
        cmd.extend(["--limit_chars", str(args.limit_chars)])
    if args.strict_background_sources:
        cmd.append("--strict_background_sources")
    return cmd


def score_cmd(args: argparse.Namespace, clean_samples: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "score_image_quality.py"),
        "--data_root",
        str(args.data_root),
        "--clean_samples",
        str(clean_samples),
        "--experiment_dir",
        str(args.out_dir),
        "--out_dir",
        str(args.out_dir / "quality_scores"),
        "--filtered_manifest_dir",
        str(args.out_dir / "quality_scores" / "filtered_manifests"),
        "--min_source_score",
        str(args.min_source_score),
        "--min_generated_score",
        str(args.min_generated_score),
        "--max_bbox_fill_ratio",
        str(args.max_bbox_fill_ratio),
    ]
    if args.require_source_ok_for_manifest:
        cmd.append("--require_source_ok_for_manifest")
    return cmd


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    args.out_dir.mkdir(parents=True, exist_ok=True)

    clean_for_generation = args.clean_samples
    if args.filter_sources_first:
        quality_dir = args.out_dir / "source_quality_prefilter"
        filtered_clean = quality_dir / "clean_samples_quality_filtered.csv"
        run(
            [
                sys.executable,
                str(ROOT / "tools" / "score_image_quality.py"),
                "--data_root",
                str(args.data_root),
                "--clean_samples",
                str(args.clean_samples),
                "--out_dir",
                str(quality_dir),
                "--filtered_clean_samples",
                str(filtered_clean),
                "--min_source_score",
                str(args.min_source_score),
                "--max_bbox_fill_ratio",
                str(args.max_bbox_fill_ratio),
            ]
        )
        clean_for_generation = filtered_clean

    simple_dir = args.out_dir / "augment" / "simple_fusion"
    run(generate_cmd(args, clean_for_generation, simple_dir, "none"))

    methods = [method.strip() for method in args.pre_extract_methods.split(",") if method.strip()]
    for method in methods:
        run(generate_cmd(args, clean_for_generation, args.out_dir / "augment" / f"preenhance_{method}", method))

    run(score_cmd(args, clean_for_generation))

    summary = {
        "out_dir": str(args.out_dir),
        "clean_samples": str(clean_for_generation),
        "pre_extract_methods": methods,
        "quality_scores": str(args.out_dir / "quality_scores"),
        "uses_cnn": False,
    }
    (args.out_dir / "quality_experiment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run no-CNN augmentation quality experiment.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--clean_samples", type=Path, required=True)
    parser.add_argument("--background_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--rare_chars", type=Path)
    parser.add_argument("--target_count", type=int, default=20)
    parser.add_argument("--limit_chars", type=int)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--pre_extract_methods", default="gamma,clahe,usm,gamma_usm,guided_gamma_usm,median_gamma_usm")
    parser.add_argument("--filter_sources_first", action="store_true")
    parser.add_argument("--require_source_ok_for_manifest", action="store_true")
    parser.add_argument("--strict_background_sources", action="store_true")
    parser.add_argument("--mask_feather", type=int, default=1)
    parser.add_argument("--ink_strength_min", type=float, default=190.0)
    parser.add_argument("--ink_strength_max", type=float, default=260.0)
    parser.add_argument("--alpha_power", type=float, default=0.45)
    parser.add_argument("--darkness_gamma", type=float, default=0.6)
    parser.add_argument("--source_bg_strength", type=float, default=0.45)
    parser.add_argument("--white_alpha_suppression", type=float, default=0.85)
    parser.add_argument("--sample_attempts", type=int, default=30)
    parser.add_argument("--max_bbox_fill_ratio", type=float, default=0.72)
    parser.add_argument("--min_source_score", type=float, default=0.38)
    parser.add_argument("--min_generated_score", type=float, default=0.36)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
