"""Extract a character region mask from a prompted character image.

This module is the first stage of the augmentation pipeline:

    prompt (box or center point) -> character region mask

It prefers a MobileSAM backend when available, but also provides an OpenCV
fallback for local smoke tests where model weights are not installed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class ExtractConfig:
    backend: str = "opencv"
    feather: int = 5
    morph_kernel: int = 3
    grabcut_iters: int = 3
    min_component_area: int = 12
    max_component_aspect: float = 18.0


@dataclass(frozen=True)
class ExtractResult:
    image: str
    backend: str
    prompt_box: tuple[int, int, int, int] | None
    prompt_point: tuple[int, int] | None
    mask_area: int
    mask_ratio: float


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    if not path.exists():
        return None
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


def parse_box(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--box must be xmin,ymin,xmax,ymax")
    x1, y1, x2, y2 = parts
    if x2 <= x1 or y2 <= y1:
        raise ValueError("--box must satisfy xmax > xmin and ymax > ymin")
    return x1, y1, x2, y2


def parse_point(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 2:
        raise ValueError("--point must be x,y")
    return parts[0], parts[1]


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def box_from_point(point: tuple[int, int], width: int, height: int, scale: float = 0.7) -> tuple[int, int, int, int]:
    side = int(min(width, height) * scale)
    side = max(16, side)
    cx, cy = point
    half = side // 2
    return clamp_box((cx - half, cy - half, cx + half, cy + half), width, height)


def largest_relevant_components(
    mask: np.ndarray,
    prompt_box: tuple[int, int, int, int] | None,
    cfg: ExtractConfig,
) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    kept = np.zeros_like(mask)
    x1 = y1 = x2 = y2 = None
    if prompt_box is not None:
        x1, y1, x2, y2 = prompt_box

    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if area < cfg.min_component_area:
            continue
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > cfg.max_component_aspect:
            continue
        if prompt_box is not None:
            cx = x + w / 2
            cy = y + h / 2
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue
        kept[labels == label_id] = 255

    return kept if np.any(kept) else mask


def feather_mask(mask: np.ndarray, feather: int) -> np.ndarray:
    if feather <= 0:
        return mask
    feather = feather if feather % 2 == 1 else feather + 1
    return cv2.GaussianBlur(mask, (feather, feather), 0)


def extract_with_opencv(
    image_bgr: np.ndarray,
    prompt_box: tuple[int, int, int, int] | None,
    prompt_point: tuple[int, int] | None,
    cfg: ExtractConfig,
) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    if prompt_box is None and prompt_point is not None:
        prompt_box = box_from_point(prompt_point, width, height)
    if prompt_box is not None:
        prompt_box = clamp_box(prompt_box, width, height)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    if prompt_box is None:
        work = gray
        offset = (0, 0)
    else:
        x1, y1, x2, y2 = prompt_box
        work = gray[y1:y2, x1:x2]
        offset = (x1, y1)

    _, otsu = cv2.threshold(work, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(work, 50, 150)
    seed = cv2.bitwise_or(otsu, edges)

    kernel = np.ones((cfg.morph_kernel, cfg.morph_kernel), np.uint8)
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, kernel, iterations=1)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, kernel, iterations=1)

    full_seed = np.zeros((height, width), dtype=np.uint8)
    ox, oy = offset
    full_seed[oy : oy + seed.shape[0], ox : ox + seed.shape[1]] = seed
    full_seed = largest_relevant_components(full_seed, prompt_box, cfg)

    if prompt_box is not None:
        grab_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
        grab_mask[full_seed > 0] = cv2.GC_PR_FGD
        x1, y1, x2, y2 = prompt_box
        prompt_slice = grab_mask[y1:y2, x1:x2]
        prompt_slice[full_seed[y1:y2, x1:x2] > 0] = cv2.GC_FGD
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (x1, y1, x2 - x1, y2 - y1)
        try:
            cv2.grabCut(image_bgr, grab_mask, rect, bgd_model, fgd_model, cfg.grabcut_iters, cv2.GC_INIT_WITH_MASK)
            full_seed = np.where((grab_mask == cv2.GC_FGD) | (grab_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        except cv2.error:
            pass

    full_seed = cv2.morphologyEx(full_seed, cv2.MORPH_CLOSE, kernel, iterations=1)
    return feather_mask(full_seed, cfg.feather)


def extract_with_mobilesam(
    image_bgr: np.ndarray,
    prompt_box: tuple[int, int, int, int] | None,
    prompt_point: tuple[int, int] | None,
    cfg: ExtractConfig,
) -> np.ndarray:
    """Placeholder MobileSAM adapter.

    The remote validation machine can replace this with a real MobileSAM
    predictor while keeping the same CLI contract. Local smoke tests should use
    --backend opencv.
    """
    raise RuntimeError(
        "MobileSAM backend is not configured locally. Use --backend opencv for smoke tests "
        "or install MobileSAM and wire the predictor in extract_with_mobilesam()."
    )


def save_alpha_cutout(image_bgr: np.ndarray, alpha: np.ndarray, out_path: Path) -> None:
    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    write_image(out_path, rgba)


def extract_region(
    image_path: Path,
    out_mask: Path,
    out_cutout: Path | None,
    prompt_box: tuple[int, int, int, int] | None,
    prompt_point: tuple[int, int] | None,
    cfg: ExtractConfig,
) -> ExtractResult:
    image_bgr = read_image(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if cfg.backend == "opencv":
        mask = extract_with_opencv(image_bgr, prompt_box, prompt_point, cfg)
    elif cfg.backend == "mobilesam":
        mask = extract_with_mobilesam(image_bgr, prompt_box, prompt_point, cfg)
    else:
        raise ValueError(f"Unsupported backend: {cfg.backend}")

    write_image(out_mask, mask)
    if out_cutout is not None:
        save_alpha_cutout(image_bgr, mask, out_cutout)

    area = int(np.count_nonzero(mask > 8))
    ratio = area / float(mask.shape[0] * mask.shape[1])
    return ExtractResult(
        image=str(image_path),
        backend=cfg.backend,
        prompt_box=prompt_box,
        prompt_point=prompt_point,
        mask_area=area,
        mask_ratio=ratio,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a prompted ancient-character region mask.")
    parser.add_argument("--image", required=True, type=Path, help="Input character image.")
    parser.add_argument("--out_mask", required=True, type=Path, help="Output grayscale mask path.")
    parser.add_argument("--out_cutout", type=Path, help="Optional RGBA cutout path.")
    parser.add_argument("--box", help="Prompt box: xmin,ymin,xmax,ymax.")
    parser.add_argument("--point", help="Prompt center point: x,y.")
    parser.add_argument("--backend", choices=["opencv", "mobilesam"], default="opencv")
    parser.add_argument("--feather", type=int, default=5)
    parser.add_argument("--morph_kernel", type=int, default=3)
    parser.add_argument("--grabcut_iters", type=int, default=3)
    parser.add_argument("--json", type=Path, help="Optional path for extraction metadata.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    box = parse_box(args.box)
    point = parse_point(args.point)
    if box is None and point is None:
        raise SystemExit("Provide either --box or --point.")

    cfg = ExtractConfig(
        backend=args.backend,
        feather=args.feather,
        morph_kernel=args.morph_kernel,
        grabcut_iters=args.grabcut_iters,
    )
    result = extract_region(args.image, args.out_mask, args.out_cutout, box, point, cfg)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
