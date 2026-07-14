"""Generate new character samples by fusing extracted glyphs with backgrounds.

This implements the left-column augmentation flow:

    prompt -> character region mask -> GrabCut/OpenCV refinement
    -> RGBA glyph -> color normalization -> different manuscript backgrounds
    -> edge feathering and illumination matching -> new character samples
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from character_region_extractor import ExtractConfig, extract_with_opencv, read_image, write_image


@dataclass(frozen=True)
class GeneratedSample:
    path: str
    char: str
    source_image: str
    background: str
    method: str
    angle: float
    scale: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def write_manifest(path: Path, rows: list[GeneratedSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(GeneratedSample.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def load_rare_chars(path: Path | None, max_count: int) -> set[str] | None:
    if path is None:
        return None
    rows = read_csv(path)
    rare = set()
    for row in rows:
        count = int(row.get("count", "0") or 0)
        if count < max_count:
            rare.add(row["char"])
    return rare


def list_background_patches(background_root: Path) -> list[Path]:
    patches = sorted(background_root.glob("*/*/*.png"))
    patches = [p for p in patches if p.parent.name == "patches"]
    if not patches:
        patches = sorted(background_root.rglob("*.png"))
    return patches


def resize_cover(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max(size / h, size / w)
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)
    rh, rw = resized.shape[:2]
    y = max(0, (rh - size) // 2)
    x = max(0, (rw - size) // 2)
    return resized[y : y + size, x : x + size].copy()


def make_glyph_alpha(image: np.ndarray, cfg: ExtractConfig) -> np.ndarray:
    h, w = image.shape[:2]
    pad_x = max(1, int(w * 0.04))
    pad_y = max(1, int(h * 0.04))
    prompt_box = (pad_x, pad_y, w - pad_x, h - pad_y)
    return extract_with_opencv(image, prompt_box, None, cfg)


def normalize_glyph_darkness(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg_pixels = gray[alpha < 8]
    if bg_pixels.size == 0:
        bg_level = float(np.percentile(gray, 90))
    else:
        bg_level = float(np.median(bg_pixels))
    bg_level = max(bg_level, 32.0)
    darkness = np.clip((bg_level - gray) / max(bg_level, 1.0), 0.0, 1.0)
    darkness = cv2.GaussianBlur(darkness, (3, 3), 0)
    return darkness


def random_affine(
    glyph_darkness: np.ndarray,
    alpha: np.ndarray,
    rng: random.Random,
    max_rotate: float,
    scale_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    h, w = alpha.shape[:2]
    angle = rng.uniform(-max_rotate, max_rotate)
    scale = rng.uniform(scale_range[0], scale_range[1])
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    tx = rng.uniform(-0.04 * w, 0.04 * w)
    ty = rng.uniform(-0.04 * h, 0.04 * h)
    matrix[:, 2] += (tx, ty)
    warped_darkness = cv2.warpAffine(glyph_darkness, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_alpha = cv2.warpAffine(alpha, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped_darkness, warped_alpha, angle, scale


def composite_on_background(
    glyph_darkness: np.ndarray,
    alpha: np.ndarray,
    background: np.ndarray,
    rng: random.Random,
    ink_strength_range: tuple[float, float],
    alpha_power: float,
    darkness_gamma: float,
) -> np.ndarray:
    size = background.shape[0]
    darkness = cv2.resize(glyph_darkness, (size, size), interpolation=cv2.INTER_AREA)
    a = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    a = cv2.GaussianBlur(a, (3, 3), 0)
    a = np.clip(a, 0.0, 1.0)
    a = np.power(a, alpha_power)
    darkness = np.power(np.clip(darkness, 0.0, 1.0), darkness_gamma)

    bg = background.astype(np.float32)
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_light = cv2.GaussianBlur(bg_gray, (0, 0), 9)
    light_scale = np.clip(local_light / max(float(np.mean(local_light)), 1.0), 0.82, 1.18)
    light_scale = light_scale[:, :, None]

    ink_strength = rng.uniform(ink_strength_range[0], ink_strength_range[1])
    tint = np.array([rng.uniform(-5, 8), rng.uniform(-4, 6), rng.uniform(-2, 5)], dtype=np.float32)
    ink = np.clip(bg - darkness[:, :, None] * ink_strength * light_scale + tint, 0, 255)
    out = bg * (1.0 - a[:, :, None]) + ink * a[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def generate_samples(
    data_root: Path,
    clean_samples: Path,
    rare_chars_csv: Path | None,
    background_root: Path,
    out_dir: Path,
    per_char: int,
    limit_chars: int | None,
    image_size: int,
    seed: int,
    mask_feather: int,
    ink_strength_range: tuple[float, float],
    alpha_power: float,
    darkness_gamma: float,
) -> list[GeneratedSample]:
    rng = random.Random(seed)
    rows = read_csv(clean_samples)
    rare = load_rare_chars(rare_chars_csv, max_count=20)
    if rare is not None:
        rows = [row for row in rows if row.get("char") in rare]

    by_char: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_char.setdefault(row["char"], []).append(row)
    chars = sorted(by_char)
    if limit_chars is not None:
        chars = chars[:limit_chars]

    backgrounds = list_background_patches(background_root)
    if not backgrounds:
        raise FileNotFoundError(f"No background patches found under {background_root}")

    cfg = ExtractConfig(backend="opencv", feather=mask_feather, morph_kernel=3, grabcut_iters=2)
    generated: list[GeneratedSample] = []
    image_out = out_dir / "images"
    debug_out = out_dir / "debug"

    for char in chars:
        samples = by_char[char]
        safe_char = f"U{ord(char[0]):04X}" if char else "unknown"
        for idx in range(per_char):
            row = rng.choice(samples)
            image_path = data_root / row["image_path"]
            image = read_image(image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            alpha = make_glyph_alpha(image, cfg)
            darkness = normalize_glyph_darkness(image, alpha)
            darkness, alpha, angle, scale = random_affine(darkness, alpha, rng, max_rotate=6.0, scale_range=(0.92, 1.08))

            bg_path = rng.choice(backgrounds)
            bg = read_image(bg_path, cv2.IMREAD_COLOR)
            if bg is None:
                continue
            bg = resize_cover(bg, image_size)
            out = composite_on_background(darkness, alpha, bg, rng, ink_strength_range, alpha_power, darkness_gamma)

            rel = Path(safe_char) / f"{safe_char}_{idx:04d}.png"
            out_path = image_out / rel
            write_image(out_path, out)

            if idx == 0:
                write_image(debug_out / f"{safe_char}_alpha.png", alpha)
                write_image(debug_out / f"{safe_char}_darkness.png", (np.clip(darkness, 0, 1) * 255).astype(np.uint8))

            generated.append(
                GeneratedSample(
                    path=str((Path("images") / rel).as_posix()),
                    char=char,
                    source_image=str(image_path),
                    background=str(bg_path),
                    method="opencv_grabcut_softmask_bg_fusion",
                    angle=round(angle, 4),
                    scale=round(scale, 4),
                )
            )

    return generated


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate fused single-character samples from clean labels and background patches.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--clean_samples", type=Path, required=True)
    parser.add_argument("--rare_chars", type=Path)
    parser.add_argument("--background_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--per_char", type=int, default=2)
    parser.add_argument("--limit_chars", type=int, help="Limit number of classes for smoke tests.")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--mask_feather", type=int, default=3, help="Character alpha feather. Smaller values keep strokes sharper.")
    parser.add_argument("--ink_strength_min", type=float, default=155.0)
    parser.add_argument("--ink_strength_max", type=float, default=225.0)
    parser.add_argument("--alpha_power", type=float, default=0.55, help="Values < 1 make alpha more solid while preserving soft edges.")
    parser.add_argument("--darkness_gamma", type=float, default=0.72, help="Values < 1 darken weak ink pixels.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows = generate_samples(
        data_root=args.data_root,
        clean_samples=args.clean_samples,
        rare_chars_csv=args.rare_chars,
        background_root=args.background_root,
        out_dir=args.out_dir,
        per_char=args.per_char,
        limit_chars=args.limit_chars,
        image_size=args.image_size,
        seed=args.seed,
        mask_feather=args.mask_feather,
        ink_strength_range=(args.ink_strength_min, args.ink_strength_max),
        alpha_power=args.alpha_power,
        darkness_gamma=args.darkness_gamma,
    )
    write_manifest(args.out_dir / "generated_samples.csv", rows)
    summary = {"generated": len(rows), "out_dir": str(args.out_dir)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
