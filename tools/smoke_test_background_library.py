"""Smoke test for inpainted background-library generation."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.gettempdir()) / "yijian_smoke_background_library"


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def make_fixture() -> tuple[Path, Path]:
    if TMP.exists():
        shutil.rmtree(TMP)
    data_root = TMP / "data"
    source = data_root / "toy" / "img"
    label = data_root / "toy" / "label"
    source.mkdir(parents=True, exist_ok=True)
    label.mkdir(parents=True, exist_ok=True)

    h, w = 256, 256
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    bg = 198 + 20 * x + 12 * np.sin(y * 18)
    image = np.dstack([bg + 6, bg + 2, bg - 8]).clip(0, 255).astype(np.uint8)
    cv2.line(image, (106, 78), (132, 180), (35, 28, 22), 9)
    cv2.line(image, (86, 125), (156, 125), (34, 29, 24), 8)
    write_image(source / "1.png", image)

    xml = """<?xml version='1.0' encoding='utf-8'?>
<annotation>
  <filename>1.png</filename>
  <object><name>test</name><bndbox><xmin>80</xmin><ymin>70</ymin><xmax>165</xmax><ymax>190</ymax></bndbox></object>
</annotation>
"""
    (label / "1.xml").write_text(xml, encoding="utf-8")
    return data_root, TMP / "out"


def main() -> int:
    data_root, out_root = make_fixture()
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "build_background_library.py"),
        "--data_root",
        str(data_root),
        "--out_root",
        str(out_root),
        "--source",
        "toy=toy",
        "--limit",
        "1",
        "--patches_per_image",
        "2",
        "--patch_size",
        "64",
        "--crop_method",
        "none",
        "--seed",
        "7",
    ]
    subprocess.run(cmd, check=True)
    manifest = out_root / "manifest.csv"
    summary = out_root / "summary.json"
    patches = list((out_root / "toy" / "patches").glob("*.png"))
    backgrounds = list((out_root / "toy" / "inpainted").glob("*.png"))
    masks = list((out_root / "toy" / "masks").glob("*.png"))
    if not manifest.exists() or not summary.exists():
        raise AssertionError("manifest or summary was not written")
    if len(patches) != 2 or len(backgrounds) != 1 or len(masks) != 1:
        raise AssertionError("unexpected output file count")
    print(f"smoke ok: backgrounds={len(backgrounds)}, patches={len(patches)}, output={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
