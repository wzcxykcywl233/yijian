"""Run the two-stage rare-character augmentation experiment.

Experiment flow:

1. Split real single-character data once.
2. Train on real data only for a baseline.
3. Generate simple background-fusion samples and retrain.
4. Select difficult classes by comparing real-only and simple-fusion
   validation metrics.
5. Discard simple-fusion samples for those classes, regenerate with each
   pre-extraction image enhancement method, retrain, and compare metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "method",
        "accuracy",
        "samples",
        "classes_in_eval",
        "real_train_samples",
        "augmented_train_samples",
        "generated_samples",
        "difficult_classes",
        "metrics_dir",
        "augment_dir",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def to_int(value: str) -> int:
    return int(float(value or 0))


def to_float(value: str) -> float:
    return float(value or 0.0)


def generated_count(augment_dir: Path) -> int:
    summary = augment_dir / "summary.json"
    if not summary.exists():
        return 0
    return int(read_json(summary).get("generated", 0))


def select_difficult_by_simple_gain(
    baseline_per_class: Path,
    simple_per_class: Path,
    out_csv: Path,
    target_count: int,
    difficulty_threshold: float,
    min_improvement: float,
    top_k: int | None,
) -> list[dict[str, object]]:
    baseline = {row["char"]: row for row in read_csv(baseline_per_class) if row.get("char")}
    candidates: list[dict[str, object]] = []
    for row in read_csv(simple_per_class):
        char = row.get("char", "")
        base = baseline.get(char)
        if not char or base is None:
            continue

        train_count = to_int(row.get("train_count", "0"))
        support = to_int(row.get("support", "0"))
        baseline_support = to_int(base.get("support", "0"))
        if support <= 0 or baseline_support <= 0 or train_count >= target_count:
            continue

        baseline_acc = to_float(base.get("accuracy", "0"))
        simple_acc = to_float(row.get("accuracy", "0"))
        acc_delta = simple_acc - baseline_acc
        no_clear_gain = acc_delta <= min_improvement
        still_low = simple_acc <= difficulty_threshold
        worsened = acc_delta < 0
        if not (worsened or (still_low and no_clear_gain)):
            continue

        reasons = []
        if worsened:
            reasons.append("worse_after_simple_fusion")
        if still_low:
            reasons.append("low_simple_fusion_accuracy")
        if no_clear_gain:
            reasons.append("no_clear_simple_fusion_gain")

        candidates.append(
            {
                "char": char,
                "count": train_count,
                "need_to_min_count": max(0, target_count - train_count),
                "baseline_accuracy": round(baseline_acc, 6),
                "simple_accuracy": round(simple_acc, 6),
                "accuracy_delta": round(acc_delta, 6),
                "support": support,
                "baseline_support": baseline_support,
                "reason": "+".join(reasons),
            }
        )

    candidates.sort(
        key=lambda row: (
            to_float(str(row["accuracy_delta"])),
            to_float(str(row["simple_accuracy"])),
            -to_int(str(row["support"])),
            str(row["char"]),
        )
    )
    if top_k is not None:
        candidates = candidates[:top_k]

    write_csv(
        out_csv,
        candidates,
        [
            "char",
            "count",
            "need_to_min_count",
            "baseline_accuracy",
            "simple_accuracy",
            "accuracy_delta",
            "support",
            "baseline_support",
            "reason",
        ],
    )
    return candidates


def train_cmd(args: argparse.Namespace, out_dir: Path, aug_manifest: Path | None = None) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "train_character_classifier.py"),
        "--data_root",
        str(args.data_root),
        "--train_csv",
        str(args.out_dir / "splits" / "train.csv"),
        "--val_csv",
        str(args.out_dir / "splits" / "val.csv"),
        "--test_csv",
        str(args.out_dir / "splits" / "test.csv"),
        "--out_dir",
        str(out_dir),
        "--backend",
        args.backend,
        "--eval_split",
        args.eval_split,
        "--image_size",
        str(args.train_image_size),
        "--difficulty_threshold",
        str(args.difficulty_threshold),
        "--target_count",
        str(args.target_count),
        "--seed",
        str(args.seed),
    ]
    if args.difficulty_top_k is not None:
        cmd.extend(["--difficulty_top_k", str(args.difficulty_top_k)])
    if args.backend == "torch_cnn":
        cmd.extend(
            [
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--learning_rate",
                str(args.learning_rate),
                "--device",
                args.device,
            ]
        )
    if aug_manifest is not None:
        cmd.extend(["--aug_manifest", str(aug_manifest)])
    return cmd


def generate_cmd(
    args: argparse.Namespace,
    rare_chars: Path,
    out_dir: Path,
    pre_extract_enhance: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "generate_character_samples.py"),
        "--data_root",
        str(args.data_root),
        "--clean_samples",
        str(args.out_dir / "splits" / "train.csv"),
        "--rare_chars",
        str(rare_chars),
        "--background_root",
        str(args.background_root),
        "--out_dir",
        str(out_dir),
        "--target_count",
        str(args.target_count),
        "--image_size",
        str(args.augment_image_size),
        "--pre_extract_enhance",
        pre_extract_enhance,
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
        "--seed",
        str(args.seed),
    ]
    if args.limit_chars is not None:
        cmd.extend(["--limit_chars", str(args.limit_chars)])
    if args.strict_background_sources:
        cmd.append("--strict_background_sources")
    return cmd


def run_experiment(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = args.out_dir / "splits"
    run(
        [
            sys.executable,
            str(ROOT / "tools" / "split_character_dataset.py"),
            "--clean_samples",
            str(args.clean_samples),
            "--out_dir",
            str(splits),
            "--val_ratio",
            str(args.val_ratio),
            "--test_ratio",
            str(args.test_ratio),
            "--min_count",
            str(args.target_count),
            "--seed",
            str(args.seed),
        ]
        + (["--limit_classes", str(args.split_limit_classes)] if args.split_limit_classes is not None else [])
    )

    rows: list[dict[str, object]] = []
    baseline_dir = args.out_dir / "models" / "baseline_real"
    run(train_cmd(args, baseline_dir))
    baseline_metrics = read_json(baseline_dir / "metrics.json")
    rows.append({"stage": "baseline", "method": "real_only", "metrics_dir": str(baseline_dir), **baseline_metrics})

    simple_aug = args.out_dir / "augment" / "simple_fusion"
    run(generate_cmd(args, splits / "train_rare_chars.csv", simple_aug, "none"))
    simple_dir = args.out_dir / "models" / "simple_fusion"
    run(train_cmd(args, simple_dir, simple_aug / "generated_samples.csv"))
    simple_metrics = read_json(simple_dir / "metrics.json")
    difficult_csv = args.out_dir / "difficult_chars_by_simple_gain.csv"
    difficult = select_difficult_by_simple_gain(
        baseline_dir / "per_class_metrics.csv",
        simple_dir / "per_class_metrics.csv",
        difficult_csv,
        args.target_count,
        args.difficulty_threshold,
        args.difficulty_min_improvement,
        args.difficulty_top_k,
    )
    difficult_classes = len(difficult)
    rows.append(
        {
            "stage": "simple_fusion",
            "method": "none",
            "metrics_dir": str(simple_dir),
            "augment_dir": str(simple_aug),
            "generated_samples": generated_count(simple_aug),
            "difficult_classes": difficult_classes,
            **simple_metrics,
        }
    )

    methods = [method.strip() for method in args.pre_extract_methods.split(",") if method.strip()]
    for method in methods:
        aug_dir = args.out_dir / "augment" / f"difficult_{method}"
        run(generate_cmd(args, difficult_csv, aug_dir, method))
        model_dir = args.out_dir / "models" / f"difficult_{method}"
        run(train_cmd(args, model_dir, aug_dir / "generated_samples.csv"))
        metrics = read_json(model_dir / "metrics.json")
        rows.append(
            {
                "stage": "difficult_reaugment",
                "method": method,
                "metrics_dir": str(model_dir),
                "augment_dir": str(aug_dir),
                "generated_samples": generated_count(aug_dir),
                "difficult_classes": difficult_classes,
                **metrics,
            }
        )

    write_summary(args.out_dir / "experiment_summary.csv", rows)
    (args.out_dir / "experiment_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline/simple/difficult re-augmentation experiment.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--clean_samples", type=Path, required=True)
    parser.add_argument("--background_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--target_count", type=int, default=20)
    parser.add_argument("--limit_chars", type=int, help="Limit generated valid classes per augmentation stage.")
    parser.add_argument("--split_limit_classes", type=int, help="Smoke-test limit on split classes.")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--backend", choices=["nearest_centroid", "torch_cnn"], default="nearest_centroid")
    parser.add_argument("--eval_split", choices=["val", "test"], default="val")
    parser.add_argument("--train_image_size", type=int, default=64)
    parser.add_argument("--augment_image_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--difficulty_threshold", type=float, default=0.6)
    parser.add_argument(
        "--difficulty_min_improvement",
        type=float,
        default=0.02,
        help="Minimum per-character validation-accuracy gain required to treat simple fusion as useful.",
    )
    parser.add_argument("--difficulty_top_k", type=int, default=20)
    parser.add_argument("--pre_extract_methods", default="gamma,clahe,usm,gamma_usm,guided_gamma_usm,median_gamma_usm")
    parser.add_argument("--strict_background_sources", action="store_true")
    parser.add_argument("--mask_feather", type=int, default=1)
    parser.add_argument("--ink_strength_min", type=float, default=190.0)
    parser.add_argument("--ink_strength_max", type=float, default=260.0)
    parser.add_argument("--alpha_power", type=float, default=0.45)
    parser.add_argument("--darkness_gamma", type=float, default=0.6)
    parser.add_argument("--source_bg_strength", type=float, default=0.45)
    parser.add_argument("--white_alpha_suppression", type=float, default=0.85)
    parser.add_argument("--sample_attempts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
