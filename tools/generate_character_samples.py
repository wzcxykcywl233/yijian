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
    char_code: str
    source_label: str
    source_image: str
    background_source: str
    background: str
    method: str
    pre_extract_enhance: str
    angle: float
    scale: float


SOURCE_LABEL_TO_BACKGROUND = {
    "天回": "tianhui",
    "张家山": "zhangjiashan",
    "武威": "wuwei",
    "马王堆": "mawangdui",
}

REQUIRED_BACKGROUND_SOURCES = ["tianhui", "zhangjiashan", "wuwei", "mawangdui"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def write_manifest(path: Path, rows: list[GeneratedSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(GeneratedSample.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def load_rare_chars(path: Path | None, max_count: int) -> list[str] | None:
    if path is None:
        return None
    rows = read_csv(path)
    rare: list[str] = []
    for row in rows:
        count = int(row.get("count", "0") or 0)
        if count < max_count:
            rare.append(row["char"])
    return rare


def safe_char_label(char: str) -> str:
    if not char:
        return "unknown"
    return "_".join(f"U{ord(ch):04X}" for ch in char)


def filesystem_char_label(char: str) -> str:
    label = (char or "").strip()
    invalid = set('<>:"/\\|?*')
    if not label or any(ch in invalid or ord(ch) < 32 for ch in label):
        return safe_char_label(char)
    label = label.rstrip(" .")
    return label or safe_char_label(char)


def list_background_patches(background_root: Path) -> list[Path]:
    patches = sorted(background_root.glob("*/*/*.png"))
    patches = [p for p in patches if p.parent.name == "patches"]
    if not patches:
        patches = sorted(background_root.rglob("*.png"))
    return patches


def background_source_name(path: Path, background_root: Path) -> str:
    try:
        return path.relative_to(background_root).parts[0]
    except ValueError:
        return path.parent.parent.name if path.parent.name == "patches" else "unknown"


def group_backgrounds_by_source(background_root: Path, paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(background_source_name(path, background_root), []).append(path)
    return grouped


def choose_background(
    by_source: dict[str, list[Path]],
    index: int,
    rng: random.Random,
    policy: str,
    strict_sources: bool,
) -> Path:
    if policy == "cycle_all_sources":
        missing = [source for source in REQUIRED_BACKGROUND_SOURCES if source not in by_source]
        if strict_sources and missing:
            raise FileNotFoundError(f"Missing required background sources: {missing}")
        cycle_sources = [source for source in REQUIRED_BACKGROUND_SOURCES if source in by_source]
        if not cycle_sources:
            cycle_sources = sorted(by_source)
        source = cycle_sources[index % len(cycle_sources)]
        return rng.choice(by_source[source])
    if policy != "random":
        raise ValueError(f"Unsupported background source policy: {policy}")
    all_paths = [path for paths in by_source.values() for path in paths]
    return rng.choice(all_paths)


def resize_cover(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max(size / h, size / w)
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)
    rh, rw = resized.shape[:2]
    y = max(0, (rh - size) // 2)
    x = max(0, (rw - size) // 2)
    return resized[y : y + size, x : x + size].copy()


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    table = np.clip(((np.arange(256, dtype=np.float32) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def apply_clahe(image: np.ndarray, clip_limit: float, tile_grid_size: int) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def apply_usm(image: np.ndarray, amount: float, radius: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def apply_edge_preserving_smooth(image: np.ndarray) -> np.ndarray:
    # OpenCV's ximgproc guided filter is optional; bilateral filtering is a
    # lightweight fallback that also suppresses texture while preserving strokes.
    ximgproc = getattr(cv2, "ximgproc", None)
    if ximgproc is not None and hasattr(ximgproc, "guidedFilter"):
        return ximgproc.guidedFilter(guide=image, src=image, radius=5, eps=80.0)
    return cv2.bilateralFilter(image, d=5, sigmaColor=22, sigmaSpace=22)


def enhance_before_extraction(
    image: np.ndarray,
    mode: str,
    gamma: float,
    clahe_clip: float,
    clahe_tile: int,
    usm_amount: float,
    usm_radius: float,
) -> np.ndarray:
    if mode == "none":
        return image
    if mode == "gamma":
        return apply_gamma(image, gamma)
    if mode == "clahe":
        return apply_clahe(image, clahe_clip, clahe_tile)
    if mode == "usm":
        return apply_usm(image, usm_amount, usm_radius)
    if mode == "gamma_clahe":
        return apply_clahe(apply_gamma(image, gamma), clahe_clip, clahe_tile)
    if mode == "gamma_usm":
        return apply_usm(apply_gamma(image, gamma), usm_amount, usm_radius)
    if mode == "guided_gamma_usm":
        return apply_usm(apply_gamma(apply_edge_preserving_smooth(image), gamma), usm_amount, usm_radius)
    if mode == "median_gamma_usm":
        return apply_usm(apply_gamma(cv2.medianBlur(image, 3), gamma), usm_amount, usm_radius)
    raise ValueError(f"Unsupported pre-extraction enhancement mode: {mode}")


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


def glyph_quality_ok(
    alpha: np.ndarray,
    darkness: np.ndarray,
    min_alpha_ratio: float,
    max_alpha_ratio: float,
    min_darkness_mean: float,
) -> bool:
    glyph = alpha > 16
    alpha_ratio = float(np.count_nonzero(glyph) / glyph.size)
    if alpha_ratio < min_alpha_ratio or alpha_ratio > max_alpha_ratio:
        return False
    if not np.any(glyph):
        return False
    return float(np.mean(darkness[glyph])) >= min_darkness_mean


def make_source_background_context(
    image: np.ndarray,
    alpha: np.ndarray,
    white_threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate usable local background from the single-character crop.

    White scan padding is deliberately assigned zero weight, so it cannot be
    treated as bamboo-slip texture during the later background pre-fusion step.
    """
    glyph = alpha > 8
    white = np.all(image >= white_threshold, axis=2)
    usable = (~glyph) & (~white)

    inpaint_mask = np.where(glyph | white, 255, 0).astype(np.uint8)
    if int(np.count_nonzero(usable)) < max(16, image.shape[0] * image.shape[1] // 200):
        context = image.copy()
        weight = np.zeros(alpha.shape, dtype=np.float32)
    else:
        context = cv2.inpaint(image, inpaint_mask, 3, cv2.INPAINT_TELEA)
        weight = usable.astype(np.float32)
        weight = cv2.GaussianBlur(weight, (0, 0), 7)
        weight = np.clip(weight / max(float(weight.max()), 1e-6), 0.0, 1.0)
    return context, weight


def match_context_to_background(context: np.ndarray, background: np.ndarray, weight: np.ndarray) -> np.ndarray:
    mask = weight > 0.08
    if int(np.count_nonzero(mask)) < 16:
        return context

    src = context.astype(np.float32)
    bg = background.astype(np.float32)
    src_mean = np.mean(src[mask], axis=0)
    bg_mean = np.mean(bg[mask], axis=0)
    matched = src + (bg_mean - src_mean) * 0.75

    src_gray = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY).astype(np.float32)
    src_std = max(float(np.std(src_gray[mask])), 1.0)
    bg_std = max(float(np.std(bg_gray[mask])), 1.0)
    contrast = np.clip(bg_std / src_std, 0.75, 1.35)
    matched = bg_mean + (matched - bg_mean) * contrast
    return np.clip(matched, 0, 255).astype(np.uint8)


def prefuse_background(
    background: np.ndarray,
    source_context: np.ndarray,
    source_weight: np.ndarray,
    strength: float,
) -> np.ndarray:
    if strength <= 0:
        return background

    size = background.shape[0]
    context = cv2.resize(source_context, (size, size), interpolation=cv2.INTER_CUBIC)
    weight = cv2.resize(source_weight, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    weight = cv2.GaussianBlur(weight, (0, 0), 5)
    weight = np.clip(weight * strength, 0.0, 0.85)
    if float(weight.max()) <= 0:
        return background

    context = match_context_to_background(context, background, weight)
    bg = background.astype(np.float32)
    ctx = context.astype(np.float32)
    fused = bg * (1.0 - weight[:, :, None]) + ctx * weight[:, :, None]
    return np.clip(fused, 0, 255).astype(np.uint8)


def random_affine(
    glyph_darkness: np.ndarray,
    alpha: np.ndarray,
    source_context: np.ndarray,
    source_weight: np.ndarray,
    rng: random.Random,
    max_rotate: float,
    scale_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
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
    warped_context = cv2.warpAffine(source_context, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    warped_weight = cv2.warpAffine(source_weight, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped_darkness, warped_alpha, warped_context, warped_weight, angle, scale


def suppress_alpha_on_white_padding(alpha: np.ndarray, source_weight: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return alpha
    support = cv2.GaussianBlur(source_weight.astype(np.float32), (0, 0), 9)
    support = np.clip(support / max(float(support.max()), 1e-6), 0.0, 1.0)
    keep = np.clip((1.0 - strength) + strength * support, 0.0, 1.0)
    return np.clip(alpha.astype(np.float32) * keep, 0, 255).astype(np.uint8)


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
    prefuse_source_bg: bool,
    source_bg_strength: float,
    source_white_threshold: int,
    white_alpha_suppression: float,
    target_count: int | None,
    background_source_policy: str,
    strict_background_sources: bool,
    pre_extract_enhance: str,
    pre_extract_gamma: float,
    clahe_clip: float,
    clahe_tile: int,
    usm_amount: float,
    usm_radius: float,
    min_alpha_ratio: float,
    max_alpha_ratio: float,
    min_darkness_mean: float,
    sample_attempts: int,
) -> list[GeneratedSample]:
    rng = random.Random(seed)
    rows = read_csv(clean_samples)
    rare = load_rare_chars(rare_chars_csv, max_count=target_count or 20)
    if rare is not None:
        rare_set = set(rare)
        rows = [row for row in rows if row.get("char") in rare_set]

    by_char: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_char.setdefault(row["char"], []).append(row)
    if rare is None:
        chars = sorted(by_char)
    else:
        chars = [char for char in rare if char in by_char]
    backgrounds = list_background_patches(background_root)
    if not backgrounds:
        raise FileNotFoundError(f"No background patches found under {background_root}")
    backgrounds_by_source = group_backgrounds_by_source(background_root, backgrounds)

    cfg = ExtractConfig(backend="opencv", feather=mask_feather, morph_kernel=3, grabcut_iters=2)
    generated: list[GeneratedSample] = []
    image_out = out_dir / "images"
    debug_out = out_dir / "debug"
    generated_char_classes = 0

    for char in chars:
        if limit_chars is not None and generated_char_classes >= limit_chars:
            break
        samples = by_char[char]
        char_code = safe_char_label(char)
        char_dir = filesystem_char_label(char)
        needed = per_char if target_count is None else max(0, target_count - len(samples))
        generated_for_char = 0
        attempts = 0
        max_attempts = max(needed, needed * sample_attempts)
        while generated_for_char < needed and attempts < max_attempts:
            attempts += 1
            row = rng.choice(samples)
            image_path = data_root / row["image_path"]
            image = read_image(image_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            extraction_image = enhance_before_extraction(
                image,
                pre_extract_enhance,
                pre_extract_gamma,
                clahe_clip,
                clahe_tile,
                usm_amount,
                usm_radius,
            )
            alpha = make_glyph_alpha(extraction_image, cfg)
            darkness = normalize_glyph_darkness(extraction_image, alpha)
            if not glyph_quality_ok(alpha, darkness, min_alpha_ratio, max_alpha_ratio, min_darkness_mean):
                continue
            source_context, source_weight = make_source_background_context(image, alpha, source_white_threshold)
            darkness, alpha, source_context, source_weight, angle, scale = random_affine(
                darkness,
                alpha,
                source_context,
                source_weight,
                rng,
                max_rotate=6.0,
                scale_range=(0.92, 1.08),
            )
            alpha = suppress_alpha_on_white_padding(alpha, source_weight, white_alpha_suppression)

            bg_path = choose_background(backgrounds_by_source, generated_for_char, rng, background_source_policy, strict_background_sources)
            bg = read_image(bg_path, cv2.IMREAD_COLOR)
            if bg is None:
                continue
            bg = resize_cover(bg, image_size)
            if prefuse_source_bg:
                bg = prefuse_background(bg, source_context, source_weight, source_bg_strength)
            out = composite_on_background(darkness, alpha, bg, rng, ink_strength_range, alpha_power, darkness_gamma)

            rel = Path(char_dir) / f"{char_dir}_{generated_for_char:04d}.png"
            out_path = image_out / rel
            write_image(out_path, out)

            if generated_for_char == 0:
                write_image(debug_out / f"{char_dir}_alpha.png", alpha)
                write_image(debug_out / f"{char_dir}_darkness.png", (np.clip(darkness, 0, 1) * 255).astype(np.uint8))
                write_image(debug_out / f"{char_dir}_source_bg_context.png", source_context)
                write_image(debug_out / f"{char_dir}_source_bg_weight.png", (np.clip(source_weight, 0, 1) * 255).astype(np.uint8))

            generated.append(
                GeneratedSample(
                    path=str((Path("images") / rel).as_posix()),
                    char=char,
                    char_code=char_code,
                    source_label=row.get("source_label", ""),
                    source_image=str(image_path),
                    background_source=background_source_name(bg_path, background_root),
                    background=str(bg_path),
                    method="preenhanced_opencv_grabcut_softmask_source_bg_prefusion"
                    if prefuse_source_bg
                    else "preenhanced_opencv_grabcut_softmask_bg_fusion",
                    pre_extract_enhance=pre_extract_enhance,
                    angle=round(angle, 4),
                    scale=round(scale, 4),
                )
            )
            generated_for_char += 1
        if generated_for_char > 0:
            generated_char_classes += 1

    return generated


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate fused single-character samples from clean labels and background patches.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--clean_samples", type=Path, required=True)
    parser.add_argument("--rare_chars", type=Path)
    parser.add_argument("--background_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--per_char", type=int, default=2)
    parser.add_argument("--target_count", type=int, help="Generate enough samples to bring each selected class up to this count.")
    parser.add_argument("--limit_chars", type=int, help="Limit number of classes for smoke tests.")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--mask_feather", type=int, default=3, help="Character alpha feather. Smaller values keep strokes sharper.")
    parser.add_argument("--ink_strength_min", type=float, default=155.0)
    parser.add_argument("--ink_strength_max", type=float, default=225.0)
    parser.add_argument("--alpha_power", type=float, default=0.55, help="Values < 1 make alpha more solid while preserving soft edges.")
    parser.add_argument("--darkness_gamma", type=float, default=0.72, help="Values < 1 darken weak ink pixels.")
    parser.add_argument("--no_prefuse_source_bg", action="store_true", help="Disable single-character local background pre-fusion.")
    parser.add_argument("--source_bg_strength", type=float, default=0.55, help="How strongly usable non-white local crop background is blended into the target background before compositing.")
    parser.add_argument("--source_white_threshold", type=int, default=245, help="Pixels with all channels above this value are treated as white scan padding.")
    parser.add_argument("--white_alpha_suppression", type=float, default=0.65, help="Suppress character alpha in areas supported only by white scan padding.")
    parser.add_argument(
        "--background_source_policy",
        choices=["cycle_all_sources", "random"],
        default="cycle_all_sources",
        help="cycle_all_sources rotates through tianhui/zhangjiashan/wuwei/mawangdui for every character.",
    )
    parser.add_argument("--strict_background_sources", action="store_true", help="Require all four manuscript background sources to exist.")
    parser.add_argument("--no_prefer_absent_source_backgrounds", action="store_true", help="Deprecated compatibility flag; use --background_source_policy random instead.")
    parser.add_argument(
        "--pre_extract_enhance",
        choices=["none", "gamma", "clahe", "usm", "gamma_clahe", "gamma_usm", "guided_gamma_usm", "median_gamma_usm"],
        default="none",
        help="Enhance the single-character image before mask/darkness extraction.",
    )
    parser.add_argument("--pre_extract_gamma", type=float, default=0.9)
    parser.add_argument("--clahe_clip", type=float, default=2.0)
    parser.add_argument("--clahe_tile", type=int, default=8)
    parser.add_argument("--usm_amount", type=float, default=0.55)
    parser.add_argument("--usm_radius", type=float, default=1.0)
    parser.add_argument("--min_alpha_ratio", type=float, default=0.003, help="Skip samples whose extracted glyph mask is too small.")
    parser.add_argument("--max_alpha_ratio", type=float, default=0.45, help="Skip samples whose extracted glyph mask is implausibly large.")
    parser.add_argument("--min_darkness_mean", type=float, default=0.015, help="Skip samples whose extracted ink signal is too weak.")
    parser.add_argument("--sample_attempts", type=int, default=20, help="Maximum attempts per requested output sample.")
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
        prefuse_source_bg=not args.no_prefuse_source_bg,
        source_bg_strength=args.source_bg_strength,
        source_white_threshold=args.source_white_threshold,
        white_alpha_suppression=args.white_alpha_suppression,
        target_count=args.target_count,
        background_source_policy="random" if args.no_prefer_absent_source_backgrounds else args.background_source_policy,
        strict_background_sources=args.strict_background_sources,
        pre_extract_enhance=args.pre_extract_enhance,
        pre_extract_gamma=args.pre_extract_gamma,
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
        usm_amount=args.usm_amount,
        usm_radius=args.usm_radius,
        min_alpha_ratio=args.min_alpha_ratio,
        max_alpha_ratio=args.max_alpha_ratio,
        min_darkness_mean=args.min_darkness_mean,
        sample_attempts=args.sample_attempts,
    )
    write_manifest(args.out_dir / "generated_samples.csv", rows)
    summary = {
        "generated": len(rows),
        "out_dir": str(args.out_dir),
        "target_count": args.target_count,
        "pre_extract_enhance": args.pre_extract_enhance,
        "background_source_policy": "random" if args.no_prefer_absent_source_backgrounds else args.background_source_policy,
        "strict_background_sources": args.strict_background_sources,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
