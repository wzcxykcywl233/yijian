"""Build a cleaned single-character label index.

The dataset contains placeholder labels such as "□" for characters that cannot
be confidently identified. These samples should not be used as classification
classes or rare-character augmentation targets, because their class labels are
unknown. This script centralizes that filtering rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_LABEL_FILES = [
    Path("天回") / "天回-单字" / "label.tsv",
    Path("张家山") / "张家山_单字" / "label.tsv",
    Path("武威") / "武威-单字" / "label.tsv",
    Path("马王堆") / "马王堆-单字" / "label.tsv",
]

DEFAULT_EXCLUDE_CHARS = ["□"]


@dataclass(frozen=True)
class LabelRow:
    source_label: str
    image_path: str
    char: str
    book: str
    page: str
    index: str


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = []
        for row in reader:
            rows.append({key: (value or "").strip() for key, value in row.items()})
        return rows


def source_name_from_label(path: Path) -> str:
    parts = path.parts
    if len(parts) >= 2:
        return parts[-3] if len(parts) >= 3 else parts[0]
    return path.stem


def build_index(
    data_root: Path,
    label_files: list[Path],
    exclude_chars: set[str],
) -> tuple[list[LabelRow], list[LabelRow], Counter[str]]:
    kept: list[LabelRow] = []
    excluded: list[LabelRow] = []
    counts: Counter[str] = Counter()

    for rel_label in label_files:
        label_path = data_root / rel_label
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        source_label = source_name_from_label(rel_label)
        for row in read_tsv(label_path):
            char = row.get("char", "")
            item = LabelRow(
                source_label=source_label,
                image_path=str((label_path.parent / source_label / row.get("path", "")).as_posix()),
                char=char,
                book=row.get("book", ""),
                page=row.get("page", ""),
                index=row.get("index", ""),
            )
            if char in exclude_chars:
                excluded.append(item)
            else:
                kept.append(item)
                counts[char] += 1
    return kept, excluded, counts


def write_rows(path: Path, rows: Iterable[LabelRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(LabelRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_counts(path: Path, counts: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["char", "count"])
        writer.writeheader()
        for char, count in sorted(counts.items(), key=lambda x: (x[1], x[0])):
            writer.writerow({"char": char, "count": count})


def write_rare(path: Path, counts: Counter[str], min_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["char", "count", "need_to_min_count"])
        writer.writeheader()
        for char, count in sorted(counts.items(), key=lambda x: (x[1], x[0])):
            if count < min_count:
                writer.writerow({"char": char, "count": count, "need_to_min_count": min_count - count})


def parse_label_files(values: list[str] | None) -> list[Path]:
    if not values:
        return DEFAULT_LABEL_FILES
    return [Path(value) for value in values]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build cleaned label index and rare-character counts.")
    parser.add_argument("--data_root", type=Path, default=Path("."), help="Dataset root.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--label_file", action="append", help="Relative label.tsv path. Can be repeated.")
    parser.add_argument("--exclude_char", action="append", default=DEFAULT_EXCLUDE_CHARS, help="Character label to exclude.")
    parser.add_argument("--min_count", type=int, default=20, help="Rare-character threshold.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    label_files = parse_label_files(args.label_file)
    exclude_chars = set(args.exclude_char or [])
    kept, excluded, counts = build_index(args.data_root, label_files, exclude_chars)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "clean_samples.csv", kept)
    write_rows(args.out_dir / "excluded_unknown_samples.csv", excluded)
    write_counts(args.out_dir / "class_counts.csv", counts)
    write_rare(args.out_dir / "rare_chars.csv", counts, args.min_count)

    rare_count = sum(1 for count in counts.values() if count < args.min_count)
    summary = {
        "label_files": [str(path) for path in label_files],
        "exclude_chars": sorted(exclude_chars),
        "kept_samples": len(kept),
        "excluded_samples": len(excluded),
        "classes": len(counts),
        "rare_classes": rare_count,
        "min_count": args.min_count,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
