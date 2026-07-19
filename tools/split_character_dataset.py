"""Create deterministic train/val/test splits for character classification."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def split_group(
    rows: list[dict[str, str]],
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    n = len(rows)
    if n <= 1:
        return rows, [], []
    if n == 2:
        return rows[:1], rows[1:], []

    val_n = max(1, round(n * val_ratio)) if val_ratio > 0 else 0
    test_n = max(1, round(n * test_ratio)) if test_ratio > 0 else 0
    while val_n + test_n >= n:
        if test_n >= val_n and test_n > 0:
            test_n -= 1
        elif val_n > 0:
            val_n -= 1
        else:
            break

    train_n = n - val_n - test_n
    return rows[:train_n], rows[train_n : train_n + val_n], rows[train_n + val_n :]


def write_rare_chars(path: Path, rows: list[dict[str, str]], min_count: int) -> None:
    counts = Counter(row["char"] for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["char", "count", "need_to_min_count"])
        writer.writeheader()
        for char, count in sorted(counts.items(), key=lambda x: (x[1], x[0])):
            if count < min_count:
                writer.writerow({"char": char, "count": count, "need_to_min_count": min_count - count})


def split_dataset(
    clean_samples: Path,
    out_dir: Path,
    val_ratio: float,
    test_ratio: float,
    min_count: int,
    seed: int,
    limit_classes: int | None,
) -> dict[str, int]:
    rows = read_csv(clean_samples)
    by_char: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("char"):
            by_char[row["char"]].append(row)

    chars = sorted(by_char)
    if limit_classes is not None:
        chars = chars[:limit_classes]

    rng = random.Random(seed)
    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    for char in chars:
        group = by_char[char][:]
        rng.shuffle(group)
        a, b, c = split_group(group, val_ratio, test_ratio)
        train.extend(a)
        val.extend(b)
        test.extend(c)

    fieldnames = list(rows[0].keys()) if rows else ["source_label", "image_path", "char", "book", "page", "index"]
    write_csv(out_dir / "train.csv", train, fieldnames)
    write_csv(out_dir / "val.csv", val, fieldnames)
    write_csv(out_dir / "test.csv", test, fieldnames)
    write_rare_chars(out_dir / "train_rare_chars.csv", train, min_count)

    summary = {
        "classes": len(chars),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "min_count": min_count,
        "seed": seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split clean character samples into train/val/test CSV files.")
    parser.add_argument("--clean_samples", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--min_count", type=int, default=20)
    parser.add_argument("--limit_classes", type=int, help="Smoke-test limit on number of classes.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = split_dataset(
        clean_samples=args.clean_samples,
        out_dir=args.out_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_count=args.min_count,
        seed=args.seed,
        limit_classes=args.limit_classes,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
