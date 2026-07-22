"""Smoke test for single-character sample generation."""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.gettempdir()) / "yijian_smoke_character_generation"


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(path)
    encoded.tofile(str(path))


def make_fixture() -> tuple[Path, Path, Path, Path]:
    if TMP.exists():
        shutil.rmtree(TMP)
    data_root = TMP / "data"
    bg_root = TMP / "backgrounds"
    out_dir = TMP / "out"
    char_dir = data_root / "toy_single"
    char_dir.mkdir(parents=True, exist_ok=True)

    img = np.full((96, 96, 3), 255, np.uint8)
    img[14:82, 18:78] = (215, 222, 230)
    for x in range(18, 78, 8):
        cv2.line(img, (x, 14), (x + 3, 81), (198, 207, 219), 1)
    cv2.line(img, (28, 18), (68, 78), (35, 28, 22), 8)
    cv2.line(img, (24, 55), (74, 55), (35, 28, 22), 7)
    cv2.GaussianBlur(img, (3, 3), 0, dst=img)
    write_image(char_dir / "char.png", img)

    bg = np.full((128, 128, 3), (190, 198, 208), np.uint8)
    for y in range(0, 128, 7):
        cv2.line(bg, (0, y), (127, y + 8), (178, 188, 200), 1)
    write_image(bg_root / "toy" / "patches" / "bg.png", bg)

    clean = TMP / "clean_samples.csv"
    with clean.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_label", "image_path", "char", "book", "page", "index"])
        writer.writeheader()
        writer.writerow({"source_label": "toy", "image_path": "toy_single/char.png", "char": "一", "book": "toy", "page": "1", "index": "1"})

    rare = TMP / "rare_chars.csv"
    with rare.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["char", "count", "need_to_min_count"])
        writer.writeheader()
        writer.writerow({"char": "一", "count": "1", "need_to_min_count": "19"})
    return data_root, clean, rare, bg_root


def main() -> int:
    data_root, clean, rare, bg_root = make_fixture()
    out_dir = TMP / "out"
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "generate_character_samples.py"),
        "--data_root",
        str(data_root),
        "--clean_samples",
        str(clean),
        "--rare_chars",
        str(rare),
        "--background_root",
        str(bg_root),
        "--out_dir",
        str(out_dir),
        "--limit_chars",
        "1",
        "--per_char",
        "2",
    ]
    subprocess.run(cmd, check=True)
    outputs = list((out_dir / "images").rglob("*.png"))
    if len(outputs) != 2:
        raise AssertionError(f"expected 2 generated images, got {len(outputs)}")
    for output in outputs:
        image = cv2.imdecode(np.fromfile(str(output), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise AssertionError(f"failed to read generated image: {output}")
        if image.shape[:2] != (96, 96):
            raise AssertionError(f"expected output size 96x96 to match source, got {image.shape[:2]} for {output}")
    if not (out_dir / "images" / "一").exists():
        raise AssertionError("character-named output directory missing")
    if not (out_dir / "generated_samples.csv").exists():
        raise AssertionError("manifest missing")
    print(f"smoke ok: generated={len(outputs)}, output={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
