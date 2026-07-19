"""Build a background library from single-character crops.

This is the preferred background source for rare-character augmentation:

    single-character crop -> glyph mask -> inpaint glyph -> background patch

Unlike full-slice background extraction, this keeps the target scale and local
material texture close to the single-character samples used by classification.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from character_region_extractor import ExtractConfig, extract_with_opencv, read_image, write_image


SOURCE_LABEL_TO_BACKGROUND = {
    "天回": "tianhui",
    "张家山": "zhangjiashan",
    "武威": "wuwei",
    "马王堆": "mawangdui",
}


@dataclass(frozen=True)
class SingleCharBackgroundRecord:
    source: str
    char: str
    char_code: str
    image: str
    background: str
    patch: str
    mask_ratio: float
    white_ratio: float
    gray_std: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def safe_char_label(char: str) -> str:
    if not char:
        return "unknown"
    return "_".join(f"U{ord(ch):04X}" for ch in char)


def resize_cover(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max(size / h, size / w)
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)
    rh, rw = resized.shape[:2]
    y = max(0, (rh - size) // 2)
    x = max(0, (rw - size) // 2)
    return resized[y : y + size, x : x + size].copy()


def glyph_mask(image: np.ndarray, feather: int, dilate: int) -> np.ndarray:
    h, w = image.shape[:2]
    pad_x = max(1, int(w * 0.04))
    pad_y = max(1, int(h * 0.04))
    box = (pad_x, pad_y, w - pad_x, h - pad_y)
    cfg = ExtractConfig(backend="opencv", feather=feather, morph_kernel=3, grabcut_iters=2)
    alpha = extract_with_opencv(image, box, None, cfg)
    mask = np.where(alpha > 8, 255, 0).astype(np.uint8)
    if dilate > 0:
        kernel_size = dilate * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def patch_stats(patch: np.ndarray) -> tuple[float, float]:
    if patch.ndim == 3:
        white_ratio = float(np.mean(np.all(patch >= 245, axis=2)))
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        white_ratio = float(np.mean(patch >= 245))
        gray = patch
    return white_ratio, float(np.std(gray))


def is_patch_usable(patch: np.ndarray, mask_ratio: float, max_mask_ratio: float, max_white_ratio: float, min_std: float) -> bool:
    white_ratio, gray_std = patch_stats(patch)
    return mask_ratio <= max_mask_ratio and white_ratio <= max_white_ratio and gray_std >= min_std


def process_row(
    data_root: Path,
    out_root: Path,
    row: dict[str, str],
    source: str,
    patch_size: int,
    feather: int,
    dilate: int,
    inpaint_radius: int,
    max_mask_ratio: float,
    max_white_ratio: float,
    min_std: float,
) -> SingleCharBackgroundRecord | None:
    image_path = data_root / row["image_path"]
    image = read_image(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return None

    mask = glyph_mask(image, feather, dilate)
    mask_ratio = float(np.count_nonzero(mask) / mask.size)
    background = cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)
    patch = resize_cover(background, patch_size)
    white_ratio, gray_std = patch_stats(patch)
    if not is_patch_usable(patch, mask_ratio, max_mask_ratio, max_white_ratio, min_std):
        return None

    safe_char = safe_char_label(row.get("char", ""))
    stem = Path(row["image_path"]).stem
    rel_name = f"{safe_char}_{stem}.png"
    source_out = out_root / source
    bg_path = source_out / "inpainted" / rel_name
    mask_path = source_out / "masks" / rel_name
    patch_path = source_out / "patches" / rel_name
    write_image(bg_path, background)
    write_image(mask_path, mask)
    write_image(patch_path, patch)

    return SingleCharBackgroundRecord(
        source=source,
        char=row.get("char", ""),
        char_code=safe_char,
        image=str(image_path),
        background=str(bg_path),
        patch=str(patch_path),
        mask_ratio=mask_ratio,
        white_ratio=white_ratio,
        gray_std=gray_std,
    )


def build_library(
    data_root: Path,
    clean_samples: Path,
    out_root: Path,
    limit_per_source: int | None,
    max_per_char: int | None,
    patch_size: int,
    feather: int,
    dilate: int,
    inpaint_radius: int,
    max_mask_ratio: float,
    max_white_ratio: float,
    min_std: float,
    seed: int,
) -> list[SingleCharBackgroundRecord]:
    rng = random.Random(seed)
    rows = read_csv(clean_samples)
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    per_char_counts: dict[tuple[str, str], int] = defaultdict(int)

    for row in rows:
        source = SOURCE_LABEL_TO_BACKGROUND.get(row.get("source_label", ""))
        if source is None:
            continue
        key = (source, row.get("char", ""))
        if max_per_char is not None and per_char_counts[key] >= max_per_char:
            continue
        per_char_counts[key] += 1
        by_source[source].append(row)

    records: list[SingleCharBackgroundRecord] = []
    for source in ("tianhui", "zhangjiashan", "wuwei", "mawangdui"):
        source_rows = by_source.get(source, [])
        rng.shuffle(source_rows)
        if limit_per_source is not None:
            source_rows = source_rows[:limit_per_source]
        source_records = []
        for row in source_rows:
            record = process_row(
                data_root,
                out_root,
                row,
                source,
                patch_size,
                feather,
                dilate,
                inpaint_radius,
                max_mask_ratio,
                max_white_ratio,
                min_std,
            )
            if record is not None:
                source_records.append(record)
        records.extend(source_records)
        print(f"{source}: {len(source_records)} patches")

    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build background patches from single-character crops.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--clean_samples", type=Path, required=True)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--limit_per_source", type=int, help="Limit processed single-character images per source.")
    parser.add_argument("--max_per_char", type=int, default=3, help="Avoid letting common classes dominate the background library.")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--feather", type=int, default=1)
    parser.add_argument("--dilate", type=int, default=2)
    parser.add_argument("--inpaint_radius", type=int, default=5)
    parser.add_argument("--max_mask_ratio", type=float, default=0.65)
    parser.add_argument("--max_white_ratio", type=float, default=0.85)
    parser.add_argument("--min_std", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)
    records = build_library(
        data_root=args.data_root,
        clean_samples=args.clean_samples,
        out_root=args.out_root,
        limit_per_source=args.limit_per_source,
        max_per_char=args.max_per_char,
        patch_size=args.patch_size,
        feather=args.feather,
        dilate=args.dilate,
        inpaint_radius=args.inpaint_radius,
        max_mask_ratio=args.max_mask_ratio,
        max_white_ratio=args.max_white_ratio,
        min_std=args.min_std,
        seed=args.seed,
    )

    manifest = args.out_root / "manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(SingleCharBackgroundRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    summary = {
        "sources": ["tianhui", "zhangjiashan", "wuwei", "mawangdui"],
        "patches": len(records),
        "manifest": str(manifest),
        "source_type": "single_character_crops",
    }
    (args.out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
