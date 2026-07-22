"""Export generated samples grouped by their original source image.

The quality workflow scores each generated image and stores the source crop in
``reference_path``. This script turns that table into visual folders where one
original character crop sits next to all augmentation results derived from it.
"""

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


def to_float(value: str, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def quality_csv_from_args(quality_csv: Path | None, quality_dir: Path | None) -> Path:
    if quality_csv is not None:
        return quality_csv
    if quality_dir is not None:
        return quality_dir / "generated_image_quality.csv"
    raise ValueError("Pass --quality_csv or --quality_dir.")


def generated_path(row: dict[str, str]) -> Path:
    return Path(row.get("abs_path", ""))


def reference_path(row: dict[str, str]) -> Path | None:
    value = row.get("reference_path", "")
    if not value:
        value = row.get("source_image", "")
    if not value:
        return None
    return Path(value)


def source_group_key(row: dict[str, str]) -> str:
    ref = reference_path(row)
    return str(ref.resolve()) if ref is not None and ref.exists() else str(ref or "")


def select_source_groups(
    grouped: dict[str, list[dict[str, str]]],
    source_limit: int,
    min_generated_per_source: int,
    sort_by: str,
) -> list[tuple[str, list[dict[str, str]]]]:
    candidates = [(key, rows) for key, rows in grouped.items() if len(rows) >= min_generated_per_source]

    def sort_key(item: tuple[str, list[dict[str, str]]]) -> tuple[float, str]:
        key, rows = item
        scores = [to_float(row.get("quality_score", "0")) for row in rows]
        if sort_by == "min_score":
            value = min(scores) if scores else 0.0
            return (value, key)
        if sort_by == "generated_count":
            return (float(len(rows)), key)
        value = sum(scores) / len(scores) if scores else 0.0
        return (value, key)

    reverse = sort_by in {"avg_score", "generated_count"}
    selected = sorted(candidates, key=sort_key, reverse=reverse)
    if source_limit > 0:
        selected = selected[:source_limit]
    return selected


def limit_rows_per_method(rows: list[dict[str, str]], per_method: int) -> list[dict[str, str]]:
    if per_method <= 0:
        return rows
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method[row.get("method", "") or "unknown"].append(row)

    selected: list[dict[str, str]] = []
    for method_rows in by_method.values():
        ordered = sorted(method_rows, key=lambda row: to_float(row.get("quality_score", "0")), reverse=True)
        selected.extend(ordered[:per_method])
    return selected


def method_rank(row: dict[str, str]) -> tuple[str, str, float, str]:
    return (
        row.get("method", "") or "unknown",
        row.get("background_source", "") or "unknown",
        -to_float(row.get("quality_score", "0")),
        row.get("abs_path", ""),
    )


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def group_label(index: int, source_key: str, rows: list[dict[str, str]]) -> str:
    first = rows[0]
    chars = sorted({row.get("char", "") for row in rows if row.get("char", "")})
    char_text = "".join(chars[:3])
    if len(chars) > 3:
        char_text += f"_and_{len(chars) - 3}_more"
    char_part = clean_name(char_text, safe_char_label(char_text))
    source_label = clean_name(first.get("source_label", "") or "unknown_source")
    origin = clean_name(Path(source_key).stem or "source_image")
    return f"source_{index:04d}__char-{char_part}__src-{source_label}__origin-{origin}"


def filter_rows(
    rows: list[dict[str, str]],
    chars: set[str],
    only_quality_ok: bool,
    only_quality_failed: bool,
) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        if chars and row.get("char", "") not in chars:
            continue
        ok = to_bool(row.get("quality_ok", ""))
        if only_quality_ok and not ok:
            continue
        if only_quality_failed and ok:
            continue
        if not row.get("reference_path", ""):
            continue
        filtered.append(row)
    return filtered


def export_lineage(
    quality_csv: Path,
    out_dir: Path,
    source_limit: int,
    per_method: int,
    min_generated_per_source: int,
    sort_by: str,
    chars: set[str],
    only_quality_ok: bool,
    only_quality_failed: bool,
) -> list[dict[str, object]]:
    rows = filter_rows(read_csv(quality_csv), chars, only_quality_ok, only_quality_failed)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = source_group_key(row)
        if key:
            grouped[key].append(row)

    records: list[dict[str, object]] = []
    selected_groups = select_source_groups(grouped, source_limit, min_generated_per_source, sort_by)
    for group_idx, (source_key, group_rows) in enumerate(selected_groups):
        selected_rows = sorted(limit_rows_per_method(group_rows, per_method), key=method_rank)
        group_dir = out_dir / group_label(group_idx, source_key, group_rows)
        group_dir.mkdir(parents=True, exist_ok=True)

        ref = Path(source_key)
        first = group_rows[0]
        ref_ext = ref.suffix or ".png"
        ref_char = clean_name(first.get("char", ""), safe_char_label(first.get("char", "")))
        ref_dst = group_dir / f"00_original__char-{ref_char}{ref_ext}"
        original_copied = copy_file(ref, ref_dst)

        group_records: list[dict[str, object]] = []
        for sample_idx, row in enumerate(selected_rows):
            gen = generated_path(row)
            ext = gen.suffix or ".png"
            method = clean_name(row.get("method", "") or "unknown")
            bg = clean_name(row.get("background_source", "") or "unknown")
            score = to_float(row.get("quality_score", "0"))
            ok = str(row.get("quality_ok", "")).lower()
            filename = f"{sample_idx + 1:02d}_generated__method-{method}__bg-{bg}__score-{score:.3f}__ok-{ok}{ext}"
            dst = group_dir / filename
            copied = copy_file(gen, dst)
            record = {
                "group": group_idx,
                "char": row.get("char", ""),
                "source_label": row.get("source_label", ""),
                "reference_path": source_key,
                "original_export_path": str(ref_dst) if original_copied else "",
                "generated_path": str(gen),
                "generated_export_path": str(dst) if copied else "",
                "copied": copied,
                "method": row.get("method", ""),
                "background_source": row.get("background_source", ""),
                "source_width": row.get("source_width", ""),
                "source_height": row.get("source_height", ""),
                "output_width": row.get("output_width", ""),
                "output_height": row.get("output_height", ""),
                "source_output_size_match": row.get("source_output_size_match", ""),
                "quality_score": row.get("quality_score", ""),
                "quality_ok": row.get("quality_ok", ""),
                "quality_reasons": row.get("quality_reasons", ""),
                "foreground_ssim": row.get("foreground_ssim", ""),
                "foreground_psnr": row.get("foreground_psnr", ""),
                "ssim": row.get("ssim", ""),
                "psnr": row.get("psnr", ""),
                "entropy": row.get("entropy", ""),
                "bbox_fill_ratio": row.get("bbox_fill_ratio", ""),
                "group_dir": str(group_dir),
            }
            records.append(record)
            group_records.append(record)

        (group_dir / "source.txt").write_text(
            "\n".join(
                [
                    f"char={first.get('char', '')}",
                    f"source_label={first.get('source_label', '')}",
                    f"reference_path={source_key}",
                    f"original_export_path={ref_dst if original_copied else ''}",
                    f"generated_images_in_group={len(selected_rows)}",
                    f"all_generated_images_from_source={len(group_rows)}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        write_csv(group_dir / "lineage.csv", group_records, LINEAGE_FIELDS)
    return records


LINEAGE_FIELDS = [
    "group",
    "char",
    "source_label",
    "reference_path",
    "original_export_path",
    "generated_path",
    "generated_export_path",
    "copied",
    "method",
    "background_source",
    "source_width",
    "source_height",
    "output_width",
    "output_height",
    "source_output_size_match",
    "quality_score",
    "quality_ok",
    "quality_reasons",
    "foreground_ssim",
    "foreground_psnr",
    "ssim",
    "psnr",
    "entropy",
    "bbox_fill_ratio",
    "group_dir",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export generated samples grouped by original source image.")
    parser.add_argument("--quality_csv", type=Path, help="Path to generated_image_quality.csv.")
    parser.add_argument("--quality_dir", type=Path, help="Directory containing generated_image_quality.csv.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--source_limit", type=int, default=80, help="Number of source-image groups to export. Use 0 for all.")
    parser.add_argument("--per_method", type=int, default=0, help="Max generated images per method in each group. Use 0 for all.")
    parser.add_argument("--min_generated_per_source", type=int, default=1)
    parser.add_argument("--sort_by", choices=["avg_score", "min_score", "generated_count"], default="generated_count")
    parser.add_argument("--char", action="append", default=[], help="Only export this character. Can be repeated.")
    parser.add_argument("--only_quality_ok", action="store_true")
    parser.add_argument("--only_quality_failed", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.only_quality_ok and args.only_quality_failed:
        raise SystemExit("--only_quality_ok and --only_quality_failed cannot be used together.")
    quality_csv = quality_csv_from_args(args.quality_csv, args.quality_dir)
    records = export_lineage(
        quality_csv=quality_csv,
        out_dir=args.out_dir,
        source_limit=args.source_limit,
        per_method=args.per_method,
        min_generated_per_source=args.min_generated_per_source,
        sort_by=args.sort_by,
        chars=set(args.char),
        only_quality_ok=args.only_quality_ok,
        only_quality_failed=args.only_quality_failed,
    )
    write_csv(args.out_dir / "source_lineage_manifest.csv", records, LINEAGE_FIELDS)
    print(
        {
            "groups": len({row["group"] for row in records}),
            "records": len(records),
            "copied": sum(1 for row in records if row["copied"]),
            "out_dir": str(args.out_dir),
            "manifest": str(args.out_dir / "source_lineage_manifest.csv"),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
