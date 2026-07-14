"""Smoke test for the prompted character region extractor."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(tempfile.gettempdir()) / "yijian_smoke_character_region"


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode image for {path}")
    encoded.tofile(str(path))


def read_gray(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def make_synthetic_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = 128, 128
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    bg = 205 + 18 * x + 10 * np.sin(y * 20)
    image = np.dstack([bg + 5, bg + 2, bg - 8]).clip(0, 255).astype(np.uint8)

    cv2.line(image, (18, 8), (24, 120), (95, 85, 70), 2)
    cv2.line(image, (75, 20), (96, 104), (36, 30, 24), 9)
    cv2.line(image, (50, 68), (104, 68), (35, 30, 25), 8)
    cv2.line(image, (82, 25), (63, 104), (40, 34, 28), 7)
    cv2.GaussianBlur(image, (3, 3), 0, dst=image)
    write_image(path, image)


def main() -> int:
    image = OUT_DIR / "synthetic_char.png"
    mask = OUT_DIR / "synthetic_char_mask.png"
    cutout = OUT_DIR / "synthetic_char_cutout.png"
    meta = OUT_DIR / "synthetic_char_meta.json"
    make_synthetic_image(image)

    cmd = [
        sys.executable,
        str(ROOT / "tools" / "character_region_extractor.py"),
        "--image",
        str(image),
        "--box",
        "40,15,112,112",
        "--out_mask",
        str(mask),
        "--out_cutout",
        str(cutout),
        "--json",
        str(meta),
        "--backend",
        "opencv",
    ]
    subprocess.run(cmd, check=True)
    mask_img = read_gray(mask)
    if mask_img is None:
        raise AssertionError("mask was not written")
    area = int(np.count_nonzero(mask_img > 8))
    if area < 100:
        raise AssertionError(f"mask area too small: {area}")
    if not cutout.exists() or not meta.exists():
        raise AssertionError("cutout or metadata was not written")
    print(f"smoke ok: mask_area={area}, output={OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
