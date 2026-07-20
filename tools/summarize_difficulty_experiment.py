"""Summarize an existing difficulty-augmentation experiment directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


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


def to_int(value: str) -> int:
    return int(float(value or 0))


def to_float(value: str) -> float:
    return float(value or 0.0)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def subset_metrics(per_class_metrics: Path, chars: set[str]) -> dict[str, object]:
    rows = [row for row in read_csv(per_class_metrics) if row.get("char", "") in chars]
    support = sum(to_int(row.get("support", "0")) for row in rows)
    correct = sum(to_int(row.get("correct", "0")) for row in rows)
    mean_accuracy = sum(to_float(row.get("accuracy", "0")) for row in rows) / len(rows) if rows else 0.0
    return {
        "difficult_eval_samples": support,
        "difficult_correct": correct,
        "difficult_accuracy": round(correct / support, 6) if support else 0.0,
        "difficult_mean_accuracy": round(mean_accuracy, 6),
    }


def generated_count(augment_dir: Path) -> int:
    summary = augment_dir / "summary.json"
    if not summary.exists():
        return 0
    return int(read_json(summary).get("generated", 0))


def manifest_count(path: Path) -> int | str:
    if not path.exists():
        return ""
    return len(read_csv(path))


def summarize(out_dir: Path) -> list[dict[str, object]]:
    difficult_csv = out_dir / "difficult_chars_by_simple_gain.csv"
    difficult_chars = {row["char"] for row in read_csv(difficult_csv) if row.get("char")}

    rows: list[dict[str, object]] = []
    stages = [("baseline", "real_only", out_dir / "models" / "baseline_real", None)]
    stages.append(("simple_fusion", "none", out_dir / "models" / "simple_fusion", out_dir / "augment" / "simple_fusion"))
    for model_dir in sorted((out_dir / "models").glob("difficult_*")):
        method = model_dir.name.removeprefix("difficult_")
        stages.append(("difficult_reaugment", method, model_dir, out_dir / "augment" / model_dir.name))

    simple_subset_accuracy = 0.0
    for stage, method, model_dir, augment_dir in stages:
        metrics_path = model_dir / "metrics.json"
        per_class_path = model_dir / "per_class_metrics.csv"
        if not metrics_path.exists() or not per_class_path.exists():
            continue
        metrics = read_json(metrics_path)
        subset = subset_metrics(per_class_path, difficult_chars)
        if stage == "simple_fusion":
            simple_subset_accuracy = to_float(str(subset["difficult_accuracy"]))
        row = {
            "stage": stage,
            "method": method,
            "accuracy": metrics.get("accuracy", ""),
            "correct": metrics.get("correct", ""),
            "samples": metrics.get("samples", ""),
            "augmented_train_samples": metrics.get("augmented_train_samples", ""),
            "generated_samples": generated_count(augment_dir) if augment_dir else "",
            "retained_simple_samples": manifest_count(out_dir / "augment" / "simple_fusion" / "non_difficult_generated_samples.csv")
            if stage == "difficult_reaugment"
            else "",
            "difficult_classes": len(difficult_chars),
            **subset,
            "difficult_accuracy_delta_vs_simple": "",
            "metrics_dir": str(model_dir),
            "augment_dir": str(augment_dir) if augment_dir else "",
        }
        if stage == "simple_fusion":
            row["difficult_accuracy_delta_vs_simple"] = 0.0
        elif stage == "difficult_reaugment":
            row["difficult_accuracy_delta_vs_simple"] = round(
                to_float(str(row["difficult_accuracy"])) - simple_subset_accuracy,
                6,
            )
        rows.append(row)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize an existing difficulty experiment.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--summary_csv", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = summarize(args.out_dir)
    summary_csv = args.summary_csv or args.out_dir / "experiment_summary_with_difficult_subset.csv"
    write_csv(
        summary_csv,
        rows,
        [
            "stage",
            "method",
            "accuracy",
            "correct",
            "samples",
            "augmented_train_samples",
            "generated_samples",
            "retained_simple_samples",
            "difficult_classes",
            "difficult_eval_samples",
            "difficult_correct",
            "difficult_accuracy",
            "difficult_mean_accuracy",
            "difficult_accuracy_delta_vs_simple",
            "metrics_dir",
            "augment_dir",
        ],
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"summary_csv={summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
