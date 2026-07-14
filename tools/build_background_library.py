"""Build an inpainted background library from full-slice images and VOC boxes.

Pipeline covered by this stage:

    full image + XML boxes -> character mask -> inpainting -> background patches
    -> optional brightness/color/texture/noise perturbation

The XML parser is intentionally tolerant because several dataset XML files have
broken character-name tags while their bndbox fields are still usable.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


DEFAULT_SOURCES = {
    "tianhui": Path("天回") / "天回-整片爬取",
    "zhangjiashan": Path("张家山") / "张家山-整片",
    "wuwei": Path("武威") / "武威_整片",
    "mawangdui": Path("马王堆") / "马王堆-整片爬取",
}


@dataclass(frozen=True)
class BackgroundRecord:
    source: str
    image: str
    xml: str
    background: str
    patches: int
    boxes: int
    mask_ratio: float
    crop_box: str


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode image for {path}")
    encoded.tofile(str(path))


def read_text_tolerant(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def parse_voc_loose(xml_path: Path) -> tuple[str | None, list[tuple[int, int, int, int]]]:
    text = read_text_tolerant(xml_path)
    filename_match = re.search(r"<filename>\s*([^<]+?)\s*</filename>", text, flags=re.I | re.S)
    filename = filename_match.group(1).strip() if filename_match else None
    boxes: list[tuple[int, int, int, int]] = []

    for block in re.findall(r"<bndbox>(.*?)</bndbox>", text, flags=re.I | re.S):
        coords = {}
        for key in ("xmin", "ymin", "xmax", "ymax"):
            match = re.search(rf"<{key}>\s*(-?\d+)\s*</{key}>", block, flags=re.I)
            if match:
                coords[key] = int(match.group(1))
        if len(coords) == 4 and coords["xmax"] > coords["xmin"] and coords["ymax"] > coords["ymin"]:
            boxes.append((coords["xmin"], coords["ymin"], coords["xmax"], coords["ymax"]))

    return filename, boxes


def clamp_box(box: tuple[int, int, int, int], width: int, height: int, pad: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def boxes_to_mask(
    shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    pad: int,
    dilate: int,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = clamp_box(box, width, height, pad)
        mask[y1:y2, x1:x2] = 255
    if dilate > 0:
        kernel_size = dilate * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def detect_pure_white_crop(
    image: np.ndarray,
    margin: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop pure-white canvas by scanning inward from four directions.

    A row/column is cropped only when every pixel in it is pure white. Scanning
    stops as soon as any non-white pixel is found.
    """
    if image.ndim == 3:
        white = np.all(image == 255, axis=2)
    else:
        white = image == 255
    height, width = white.shape[:2]

    top = 0
    while top < height and bool(np.all(white[top, :])):
        top += 1

    bottom = height
    while bottom > top and bool(np.all(white[bottom - 1, :])):
        bottom -= 1

    left = 0
    while left < width and bool(np.all(white[:, left])):
        left += 1

    right = width
    while right > left and bool(np.all(white[:, right - 1])):
        right -= 1

    if top >= bottom or left >= right:
        return image, (0, 0, width, height)

    x1 = max(0, left - margin)
    y1 = max(0, top - margin)
    x2 = min(width, right + margin)
    y2 = min(height, bottom + margin)
    if x2 <= x1 or y2 <= y1:
        return image, (0, 0, width, height)
    return image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def detect_edge_crop(
    image: np.ndarray,
    margin: int,
    canny_low: int,
    canny_high: int,
    edge_dilate: int,
    min_row_edge_pixels: int | None = None,
    min_col_edge_pixels: int | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop canvas by finding the bounding box of meaningful edge projections."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    height, width = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    if edge_dilate > 0:
        kernel_size = edge_dilate * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    row_counts = np.count_nonzero(edges, axis=1)
    col_counts = np.count_nonzero(edges, axis=0)
    row_min = min_row_edge_pixels if min_row_edge_pixels is not None else max(2, int(width * 0.015))
    col_min = min_col_edge_pixels if min_col_edge_pixels is not None else max(3, int(height * 0.002))
    rows = np.flatnonzero(row_counts >= row_min)
    cols = np.flatnonzero(col_counts >= col_min)

    if rows.size == 0 or cols.size == 0:
        return detect_pure_white_crop(image, margin)

    y1 = max(0, int(rows[0]) - margin)
    y2 = min(height, int(rows[-1]) + 1 + margin)
    x1 = max(0, int(cols[0]) - margin)
    x2 = min(width, int(cols[-1]) + 1 + margin)
    if x2 <= x1 or y2 <= y1:
        return image, (0, 0, width, height)
    return image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def shift_boxes_to_crop(
    boxes: list[tuple[int, int, int, int]],
    crop_box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> list[tuple[int, int, int, int]]:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    shifted: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in boxes:
        nx1 = max(0, min(width, x1 - crop_x1))
        ny1 = max(0, min(height, y1 - crop_y1))
        nx2 = max(0, min(width, x2 - crop_x1))
        ny2 = max(0, min(height, y2 - crop_y1))
        if nx2 > nx1 and ny2 > ny1:
            shifted.append((nx1, ny1, nx2, ny2))
    return shifted


def inpaint_background(image: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    if image.ndim == 2:
        return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)
    return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)


def enhance_background_patch(patch: np.ndarray, rng: random.Random) -> np.ndarray:
    out = patch.astype(np.float32)
    alpha = rng.uniform(0.88, 1.16)
    beta = rng.uniform(-14, 14)
    out = out * alpha + beta
    out = np.clip(out, 0, 255).astype(np.uint8)

    if out.ndim == 3 and out.shape[2] == 3:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.randint(-4, 4)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + rng.randint(-10, 12), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + rng.randint(-8, 8), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if rng.random() < 0.45:
        noise = rng.normalvariate(0, 1)
        sigma = abs(noise) * 4 + 1.5
        n = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, sigma, out.shape)
        out = np.clip(out.astype(np.float32) + n, 0, 255).astype(np.uint8)

    if rng.random() < 0.35:
        blurred = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 0.9))
        out = cv2.addWeighted(out, 1.25, blurred, -0.25, 0)
    elif rng.random() < 0.25:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    return out


def crop_patches(
    background: np.ndarray,
    patch_size: int,
    patches_per_image: int,
    rng: random.Random,
    enhance: bool,
) -> list[np.ndarray]:
    height, width = background.shape[:2]
    if height <= 0 or width <= 0:
        return []

    patches: list[np.ndarray] = []
    for _ in range(patches_per_image):
        if height < patch_size or width < patch_size:
            scale = max(patch_size / height, patch_size / width)
            resized = cv2.resize(background, (int(width * scale) + 1, int(height * scale) + 1), interpolation=cv2.INTER_CUBIC)
            work = resized
            wh, ww = work.shape[:2]
        else:
            work = background
            wh, ww = height, width
        x = rng.randint(0, max(0, ww - patch_size))
        y = rng.randint(0, max(0, wh - patch_size))
        patch = work[y : y + patch_size, x : x + patch_size].copy()
        if enhance:
            patch = enhance_background_patch(patch, rng)
        patches.append(patch)
    return patches


def find_image_for_xml(source_dir: Path, xml_path: Path, filename: str | None) -> Path | None:
    img_dir = source_dir / "img"
    candidates: list[Path] = []
    if filename:
        candidates.append(img_dir / filename)
    candidates.extend([img_dir / f"{xml_path.stem}.bmp", img_dir / f"{xml_path.stem}.png", img_dir / f"{xml_path.stem}.jpg"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def process_source(
    data_root: Path,
    out_root: Path,
    source_name: str,
    source_rel: Path,
    limit: int | None,
    patch_size: int,
    patches_per_image: int,
    mask_pad: int,
    dilate: int,
    inpaint_radius: int,
    enhance: bool,
    crop_method: str,
    crop_margin: int,
    canny_low: int,
    canny_high: int,
    edge_dilate: int,
    rng: random.Random,
) -> list[BackgroundRecord]:
    source_dir = data_root / source_rel
    label_dir = source_dir / "label"
    if not label_dir.exists():
        raise FileNotFoundError(f"Missing label directory: {label_dir}")

    records: list[BackgroundRecord] = []
    xml_paths = sorted(label_dir.glob("*.xml"))
    if limit is not None:
        xml_paths = xml_paths[:limit]

    for xml_path in xml_paths:
        filename, boxes = parse_voc_loose(xml_path)
        if not boxes:
            continue
        image_path = find_image_for_xml(source_dir, xml_path, filename)
        if image_path is None:
            continue
        image = read_image(image_path, cv2.IMREAD_COLOR)
        if image is None:
            continue

        crop_box = (0, 0, image.shape[1], image.shape[0])
        if crop_method == "pure_white":
            image, crop_box = detect_pure_white_crop(image, crop_margin)
        elif crop_method == "edge":
            image, crop_box = detect_edge_crop(image, crop_margin, canny_low, canny_high, edge_dilate)
        elif crop_method == "none":
            pass
        else:
            raise ValueError(f"Unsupported crop method: {crop_method}")
        if crop_method != "none":
            boxes = shift_boxes_to_crop(boxes, crop_box, image.shape[1], image.shape[0])
            if not boxes:
                continue

        mask = boxes_to_mask(image.shape[:2], boxes, mask_pad, dilate)
        background = inpaint_background(image, mask, inpaint_radius)
        source_out = out_root / source_name
        bg_path = source_out / "inpainted" / f"{xml_path.stem}.png"
        mask_path = source_out / "masks" / f"{xml_path.stem}.png"
        write_image(bg_path, background)
        write_image(mask_path, mask)

        patch_count = 0
        for idx, patch in enumerate(crop_patches(background, patch_size, patches_per_image, rng, enhance)):
            patch_path = source_out / "patches" / f"{xml_path.stem}_{idx:03d}.png"
            write_image(patch_path, patch)
            patch_count += 1

        records.append(
            BackgroundRecord(
                source=source_name,
                image=str(image_path),
                xml=str(xml_path),
                background=str(bg_path),
                patches=patch_count,
                boxes=len(boxes),
                mask_ratio=float(np.count_nonzero(mask) / mask.size),
                crop_box=",".join(str(v) for v in crop_box),
            )
        )

    return records


def parse_sources(values: list[str] | None) -> dict[str, Path]:
    if not values:
        return DEFAULT_SOURCES
    selected: dict[str, Path] = {}
    for value in values:
        if "=" in value:
            name, rel = value.split("=", 1)
            selected[name] = Path(rel)
        elif value in DEFAULT_SOURCES:
            selected[value] = DEFAULT_SOURCES[value]
        else:
            raise ValueError(f"Unknown source '{value}'. Use one of {sorted(DEFAULT_SOURCES)} or name=relative_path.")
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build inpainted background patches from full-slice XML annotations.")
    parser.add_argument("--data_root", type=Path, default=Path("."), help="Dataset root containing the four source folders.")
    parser.add_argument("--out_root", type=Path, required=True, help="Output background library directory.")
    parser.add_argument("--source", action="append", help="Source name or name=relative_path. Can be repeated.")
    parser.add_argument("--limit", type=int, help="Limit XML files per source for smoke tests.")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--patches_per_image", type=int, default=4)
    parser.add_argument("--mask_pad", type=int, default=4)
    parser.add_argument("--dilate", type=int, default=2)
    parser.add_argument("--inpaint_radius", type=int, default=5)
    parser.add_argument("--no_enhance", action="store_true", help="Disable brightness/color/noise/texture perturbation.")
    parser.add_argument(
        "--crop_method",
        choices=["edge", "pure_white", "none"],
        default="edge",
        help="Canvas crop method. edge uses Canny projections; pure_white crops only all-white rows/columns.",
    )
    parser.add_argument("--no_crop_white_border", action="store_true", help="Deprecated alias for --crop_method none.")
    parser.add_argument("--crop_margin", type=int, default=32, help="Margin kept around detected manuscript content.")
    parser.add_argument("--canny_low", type=int, default=40)
    parser.add_argument("--canny_high", type=int, default=120)
    parser.add_argument("--edge_dilate", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rng = random.Random(args.seed)
    sources = parse_sources(args.source)
    crop_method = "none" if args.no_crop_white_border else args.crop_method
    args.out_root.mkdir(parents=True, exist_ok=True)

    all_records: list[BackgroundRecord] = []
    for source_name, source_rel in sources.items():
        records = process_source(
            data_root=args.data_root,
            out_root=args.out_root,
            source_name=source_name,
            source_rel=source_rel,
            limit=args.limit,
            patch_size=args.patch_size,
            patches_per_image=args.patches_per_image,
            mask_pad=args.mask_pad,
            dilate=args.dilate,
            inpaint_radius=args.inpaint_radius,
            enhance=not args.no_enhance,
            crop_method=crop_method,
            crop_margin=args.crop_margin,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            edge_dilate=args.edge_dilate,
            rng=rng,
        )
        all_records.extend(records)
        print(f"{source_name}: {len(records)} backgrounds")

    manifest = args.out_root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(BackgroundRecord.__dataclass_fields__.keys()))
        writer.writeheader()
        for record in all_records:
            writer.writerow(asdict(record))

    summary = {
        "sources": list(sources.keys()),
        "backgrounds": len(all_records),
        "patches": sum(record.patches for record in all_records),
        "manifest": str(manifest),
    }
    (args.out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
