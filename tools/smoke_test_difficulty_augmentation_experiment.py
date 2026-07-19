"""Smoke test for the full difficulty-driven augmentation experiment."""

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
TMP = Path(tempfile.gettempdir()) / "yijian_smoke_difficulty_experiment"


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(path)
    encoded.tofile(str(path))


def make_char_image(char_idx: int, sample_idx: int) -> np.ndarray:
    rng = np.random.default_rng(char_idx * 1000 + sample_idx)
    image = np.full((96, 96, 3), 238, dtype=np.uint8)
    image += rng.normal(0, 4, image.shape).astype(np.int16).clip(-12, 12).astype(np.uint8)
    cv2.rectangle(image, (12, 12), (84, 84), (210, 216, 222), -1)
    color = (35, 30, 28)
    if char_idx == 0:
        cv2.line(image, (30, 22), (66, 74), color, 7)
        cv2.line(image, (22, 54), (74, 54), color, 6)
    elif char_idx == 1:
        cv2.line(image, (24, 30), (74, 30), color, 6)
        cv2.line(image, (48, 30), (40, 74), color, 7)
    else:
        cv2.line(image, (24, 28), (70, 66), color, 7)
        cv2.line(image, (68, 28), (24, 70), color, 5)
    return cv2.GaussianBlur(image, (3, 3), 0)


def make_background(source_idx: int, sample_idx: int) -> np.ndarray:
    base_colors = [(190, 205, 212), (164, 145, 108), (182, 190, 180), (214, 208, 196)]
    image = np.full((128, 128, 3), base_colors[source_idx], dtype=np.uint8)
    for x in range(0, 128, 9 + source_idx):
        cv2.line(image, (x, 0), (x + sample_idx + 2, 127), tuple(max(0, c - 22) for c in base_colors[source_idx]), 1)
    return image


def make_fixture() -> tuple[Path, Path, Path]:
    if TMP.exists():
        shutil.rmtree(TMP)
    data_root = TMP / "data"
    bg_root = TMP / "backgrounds"
    chars = ["甲", "乙", "丙"]
    source_labels = ["天回", "张家山", "武威", "马王堆"]

    rows = []
    for char_idx, char in enumerate(chars):
        for sample_idx in range(6):
            source_label = source_labels[sample_idx % len(source_labels)]
            rel = Path("single") / source_label / char / f"{sample_idx}.png"
            write_image(data_root / rel, make_char_image(char_idx, sample_idx))
            rows.append(
                {
                    "source_label": source_label,
                    "image_path": rel.as_posix(),
                    "char": char,
                    "book": "smoke",
                    "page": str(sample_idx),
                    "index": str(sample_idx),
                }
            )

    clean = TMP / "clean_samples.csv"
    with clean.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_label", "image_path", "char", "book", "page", "index"])
        writer.writeheader()
        writer.writerows(rows)

    for source_idx, source in enumerate(["tianhui", "zhangjiashan", "wuwei", "mawangdui"]):
        for sample_idx in range(2):
            write_image(bg_root / source / "patches" / f"{sample_idx}.png", make_background(source_idx, sample_idx))

    return data_root, clean, bg_root


def main() -> int:
    data_root, clean, bg_root = make_fixture()
    out_dir = TMP / "experiment"
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "run_difficulty_augmentation_experiment.py"),
        "--data_root",
        str(data_root),
        "--clean_samples",
        str(clean),
        "--background_root",
        str(bg_root),
        "--out_dir",
        str(out_dir),
        "--target_count",
        "6",
        "--backend",
        "nearest_centroid",
        "--difficulty_top_k",
        "1",
        "--pre_extract_methods",
        "gamma_usm",
        "--strict_background_sources",
        "--seed",
        "7",
    ]
    subprocess.run(cmd, check=True)
    summary = out_dir / "experiment_summary.csv"
    if not summary.exists():
        raise AssertionError("experiment summary missing")
    text = summary.read_text(encoding="utf-8-sig")
    for expected in ("baseline", "simple_fusion", "difficult_reaugment"):
        if expected not in text:
            raise AssertionError(f"missing stage: {expected}")
    print(f"smoke ok: summary={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
