"""Export high-score and low-score image examples for visual comparison."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
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


def safe_char_label(char: str) -> str:
    if not char:
        return "unknown"
    return "_".join(f"U{ord(ch):04X}" for ch in char)


def clean_name(text: str, fallback: str = "unknown") -> str:
    label = (text or "").strip() or fallback
    invalid = set('<>:"/\\|?*')
    label = "".join("_" if ch in invalid or ord(ch) < 32 else ch for ch in label)
    label = label.rstrip(" .")
    return label or fallback


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def row_group(row: dict[str, str], kind: str, group_by: str) -> str:
    if group_by != "auto":
        return row.get(group_by, "") or "unknown"
    if kind == "generated":
        return row.get("method", "") or "unknown"
    return row.get("source_label", "") or "unknown"


def source_path(row: dict[str, str]) -> Path:
    return Path(row.get("abs_path", ""))


def copy_row_image(row: dict[str, str], dst: Path) -> bool:
    src = source_path(row)
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def select_examples(rows: list[dict[str, str]], per_group: int, side: str) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: to_float(row.get("quality_score", "0")), reverse=(side == "high"))
    return ordered[:per_group]


def export_examples(
    input_csv: Path,
    out_dir: Path,
    kind: str,
    per_group: int,
    group_by: str,
    include_only_failed_low: bool,
) -> list[dict[str, object]]:
    rows = read_csv(input_csv)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row_group(row, kind, group_by)].append(row)

    records: list[dict[str, object]] = []
    for group, group_rows in sorted(grouped.items()):
        high_rows = select_examples(group_rows, per_group, "high")
        low_source = [row for row in group_rows if row.get("quality_ok", "").lower() == "false"] if include_only_failed_low else group_rows
        low_rows = select_examples(low_source, per_group, "low")

        for side, selected in [("high", high_rows), ("low", low_rows)]:
            for idx, row in enumerate(selected):
                char = row.get("char", "")
                char_label = clean_name(char, safe_char_label(char))
                score = to_float(row.get("quality_score", "0"))
                reason = clean_name(row.get("quality_reasons", "") or "ok")
                ext = source_path(row).suffix or ".png"
                filename = f"{side}_{idx:04d}__score-{score:.3f}__char-{char_label}__reason-{reason}{ext}"
                dst = out_dir / kind / clean_name(group) / side / filename
                copied = copy_row_image(row, dst)
                records.append(
                    {
                        "kind": kind,
                        "group": group,
                        "side": side,
                        "rank": idx,
                        "char": char,
                        "quality_score": row.get("quality_score", ""),
                        "quality_ok": row.get("quality_ok", ""),
                        "quality_reasons": row.get("quality_reasons", ""),
                        "source_path": row.get("abs_path", ""),
                        "export_path": str(dst) if copied else "",
                        "copied": copied,
                        "method": row.get("method", ""),
                        "source_label": row.get("source_label", ""),
                        "background_source": row.get("background_source", ""),
                        "entropy": row.get("entropy", ""),
                        "ssim": row.get("ssim", ""),
                        "psnr": row.get("psnr", ""),
                        "laplacian_var": row.get("laplacian_var", ""),
                        "contrast_std": row.get("contrast_std", ""),
                        "bbox_fill_ratio": row.get("bbox_fill_ratio", ""),
                    }
                )
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export high/low quality-score image examples.")
    parser.add_argument("--quality_dir", type=Path, required=True, help="Directory containing source/generated image quality CSVs.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--kind", choices=["source", "generated", "both"], default="both")
    parser.add_argument("--per_group", type=int, default=12)
    parser.add_argument("--group_by", default="auto", help="CSV column to group by, or auto.")
    parser.add_argument("--include_only_failed_low", action="store_true", help="Low-score side only samples quality_ok=false rows.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    if args.kind in ("source", "both"):
        source_csv = args.quality_dir / "source_image_quality.csv"
        if source_csv.exists():
            records.extend(export_examples(source_csv, args.out_dir, "source", args.per_group, args.group_by, args.include_only_failed_low))
    if args.kind in ("generated", "both"):
        generated_csv = args.quality_dir / "generated_image_quality.csv"
        if generated_csv.exists():
            records.extend(export_examples(generated_csv, args.out_dir, "generated", args.per_group, args.group_by, args.include_only_failed_low))

    fieldnames = [
        "kind",
        "group",
        "side",
        "rank",
        "char",
        "quality_score",
        "quality_ok",
        "quality_reasons",
        "source_path",
        "export_path",
        "copied",
        "method",
        "source_label",
        "background_source",
        "entropy",
        "ssim",
        "psnr",
        "laplacian_var",
        "contrast_std",
        "bbox_fill_ratio",
    ]
    write_csv(args.out_dir / "quality_score_examples_manifest.csv", records, fieldnames)
    print(
        {
            "records": len(records),
            "copied": sum(1 for row in records if row["copied"]),
            "out_dir": str(args.out_dir),
            "manifest": str(args.out_dir / "quality_score_examples_manifest.csv"),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
