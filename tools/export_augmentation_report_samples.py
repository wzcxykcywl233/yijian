"""Export sampled original and augmented character images for reports.

The output is organized by character, with separate subfolders for original
images, simple background fusion, and each pre-enhancement method.
"""

from __future__ import annotations

import argparse
import csv
import random
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


def filesystem_label(text: str, fallback: str) -> str:
    label = (text or "").strip()
    invalid = set('<>:"/\\|?*')
    if not label or any(ch in invalid or ord(ch) < 32 for ch in label):
        return fallback
    label = label.rstrip(" .")
    return label or fallback


def clean_name(text: str, fallback: str = "unknown") -> str:
    label = (text or "").strip() or fallback
    invalid = set('<>:"/\\|?*')
    label = "".join("_" if ch in invalid or ord(ch) < 32 else ch for ch in label)
    label = label.rstrip(" .")
    return label or fallback


def group_by_char(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        char = row.get("char", "")
        if char:
            grouped[char].append(row)
    return grouped


def sampled(rows: list[dict[str, str]], limit: int, rng: random.Random) -> list[dict[str, str]]:
    if limit <= 0 or len(rows) <= limit:
        return rows[:]
    return rng.sample(rows, limit)


def manifest_rows(manifest: Path) -> list[dict[str, str]]:
    rows = read_csv(manifest)
    for row in rows:
        row["_manifest"] = str(manifest)
        row["_manifest_root"] = str(manifest.parent)
    return rows


def discover_augmented_manifests(experiment_dir: Path, methods: set[str] | None) -> list[tuple[str, str, Path]]:
    manifests: list[tuple[str, str, Path]] = []
    simple = experiment_dir / "augment" / "simple_fusion" / "generated_samples.csv"
    if simple.exists():
        manifests.append(("01_simple_fusion", "simple_fusion", simple))

    for manifest in sorted((experiment_dir / "augment").glob("difficult_*/generated_samples.csv")):
        method = manifest.parent.name.removeprefix("difficult_")
        if methods is not None and method not in methods:
            continue
        manifests.append((f"02_{clean_name(method)}", method, manifest))
    return manifests


def collect_target_chars(
    clean_samples: Path,
    manifests: list[tuple[str, str, Path]],
    chars_csv: Path | None,
) -> list[str]:
    if chars_csv is not None:
        rows = read_csv(chars_csv)
        return sorted({row["char"] for row in rows if row.get("char")})

    chars: set[str] = set()
    for _, _, manifest in manifests:
        for row in read_csv(manifest):
            if row.get("char"):
                chars.add(row["char"])
    if not chars:
        chars.update(row["char"] for row in read_csv(clean_samples) if row.get("char"))
    return sorted(chars)


def copy_sample(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def original_source_path(data_root: Path, row: dict[str, str]) -> Path:
    return data_root / row.get("image_path", "")


def augmented_source_path(row: dict[str, str]) -> Path:
    return Path(row["_manifest_root"]) / row.get("path", "")


def export_samples(
    data_root: Path,
    clean_samples: Path,
    experiment_dir: Path,
    out_dir: Path,
    per_group: int,
    seed: int,
    chars_csv: Path | None,
    methods: set[str] | None,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    manifests = discover_augmented_manifests(experiment_dir, methods)
    chars = collect_target_chars(clean_samples, manifests, chars_csv)

    originals = group_by_char(read_csv(clean_samples))
    augmented = [(folder, method, group_by_char(manifest_rows(manifest))) for folder, method, manifest in manifests]

    records: list[dict[str, object]] = []
    for char in chars:
        char_code = safe_char_label(char)
        char_dir = out_dir / filesystem_label(char, char_code)
        (char_dir / "label.txt").write_text(f"char={char}\nchar_code={char_code}\n", encoding="utf-8")

        original_rows = sampled(originals.get(char, []), per_group, rng)
        for idx, row in enumerate(original_rows):
            src = original_source_path(data_root, row)
            ext = src.suffix or ".png"
            dst = char_dir / "00_original" / f"original_{idx:04d}__source-{clean_name(row.get('source_label', 'unknown'))}{ext}"
            copied = copy_sample(src, dst)
            records.append(
                {
                    "char": char,
                    "char_code": char_code,
                    "group": "original",
                    "method": "real",
                    "copied": copied,
                    "source_path": str(src),
                    "export_path": str(dst) if copied else "",
                    "background_source": "",
                    "source_label": row.get("source_label", ""),
                }
            )

        for folder, method, by_char in augmented:
            rows = sampled(by_char.get(char, []), per_group, rng)
            for idx, row in enumerate(rows):
                src = augmented_source_path(row)
                ext = src.suffix or ".png"
                bg = clean_name(row.get("background_source", "unknown"))
                source_label = clean_name(row.get("source_label", "unknown"))
                dst = char_dir / folder / f"{clean_name(method)}_{idx:04d}__bg-{bg}__src-{source_label}{ext}"
                copied = copy_sample(src, dst)
                records.append(
                    {
                        "char": char,
                        "char_code": char_code,
                        "group": folder,
                        "method": method,
                        "copied": copied,
                        "source_path": str(src),
                        "export_path": str(dst) if copied else "",
                        "background_source": row.get("background_source", ""),
                        "source_label": row.get("source_label", ""),
                    }
                )

    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export sampled augmentation examples grouped by character.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--clean_samples", type=Path, required=True)
    parser.add_argument("--experiment_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--per_group", type=int, default=4, help="Random samples per character and group.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chars_csv", type=Path, help="Optional CSV containing a char column, e.g. difficult_chars_by_simple_gain.csv.")
    parser.add_argument("--methods", help="Comma-separated difficult methods to include. Default: all discovered methods.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    methods = {item.strip() for item in args.methods.split(",") if item.strip()} if args.methods else None
    records = export_samples(
        data_root=args.data_root,
        clean_samples=args.clean_samples,
        experiment_dir=args.experiment_dir,
        out_dir=args.out_dir,
        per_group=args.per_group,
        seed=args.seed,
        chars_csv=args.chars_csv,
        methods=methods,
    )
    write_csv(
        args.out_dir / "export_manifest.csv",
        records,
        ["char", "char_code", "group", "method", "copied", "source_path", "export_path", "background_source", "source_label"],
    )
    copied = sum(1 for row in records if row["copied"])
    print(
        {
            "chars": len({row["char"] for row in records}),
            "records": len(records),
            "copied": copied,
            "out_dir": str(args.out_dir),
            "manifest": str(args.out_dir / "export_manifest.csv"),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
