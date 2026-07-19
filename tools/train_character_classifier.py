"""Train/evaluate a character classifier and export per-class metrics.

The default backend is a dependency-light nearest-centroid classifier for local
smoke tests. Use ``--backend torch_cnn`` on the remote training machine when
PyTorch is installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from character_region_extractor import read_image


@dataclass(frozen=True)
class Sample:
    path: Path
    char: str
    origin: str


@dataclass(frozen=True)
class ClassMetric:
    char: str
    support: int
    correct: int
    accuracy: float
    train_count: int


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def write_dict_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_manifest_samples(manifest: Path) -> list[Sample]:
    rows = read_csv(manifest)
    root = manifest.parent
    samples = []
    for row in rows:
        char = row.get("char", "")
        rel = row.get("path", "")
        if char and rel:
            samples.append(Sample(root / rel, char, "augmented"))
    return samples


def load_split_samples(data_root: Path, csv_path: Path, origin: str) -> list[Sample]:
    samples = []
    for row in read_csv(csv_path):
        char = row.get("char", "")
        rel = row.get("image_path", "")
        if char and rel:
            samples.append(Sample(data_root / rel, char, origin))
    return samples


def load_image_vector(path: Path, image_size: int) -> np.ndarray | None:
    image = read_image(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return image.reshape(-1)


def prepare_arrays(samples: list[Sample], class_to_idx: dict[str, int], image_size: int) -> tuple[np.ndarray, np.ndarray, list[Sample]]:
    xs = []
    ys = []
    kept = []
    for sample in samples:
        if sample.char not in class_to_idx:
            continue
        vector = load_image_vector(sample.path, image_size)
        if vector is None:
            continue
        xs.append(vector)
        ys.append(class_to_idx[sample.char])
        kept.append(sample)
    if not xs:
        return np.empty((0, image_size * image_size), dtype=np.float32), np.empty((0,), dtype=np.int64), kept
    return np.stack(xs), np.asarray(ys, dtype=np.int64), kept


def predict_nearest_centroid(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, class_count: int) -> np.ndarray:
    centroids = np.zeros((class_count, train_x.shape[1]), dtype=np.float32)
    for class_idx in range(class_count):
        class_x = train_x[train_y == class_idx]
        if class_x.size:
            centroids[class_idx] = np.mean(class_x, axis=0)
    x_norm = np.sum(eval_x * eval_x, axis=1, keepdims=True)
    c_norm = np.sum(centroids * centroids, axis=1)[None, :]
    distances = x_norm + c_norm - 2.0 * eval_x @ centroids.T
    return np.argmin(distances, axis=1)


def train_torch_cnn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    image_size: int,
    class_count: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str,
) -> np.ndarray:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed. Use --backend nearest_centroid for local smoke tests.") from exc

    torch.manual_seed(seed)
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    class SmallCnn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 24, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(48, 96, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(96, class_count),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x)

    train_tensor = torch.from_numpy(train_x.reshape(-1, 1, image_size, image_size))
    label_tensor = torch.from_numpy(train_y)
    loader = DataLoader(TensorDataset(train_tensor, label_tensor), batch_size=batch_size, shuffle=True)
    model = SmallCnn().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(eval_x), batch_size):
            batch = torch.from_numpy(eval_x[start : start + batch_size].reshape(-1, 1, image_size, image_size)).to(device)
            preds.append(torch.argmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.empty((0,), dtype=np.int64)


def compute_metrics(
    eval_samples: list[Sample],
    truth: np.ndarray,
    pred: np.ndarray,
    idx_to_class: list[str],
    train_counts: Counter[str],
) -> tuple[dict[str, object], list[ClassMetric]]:
    total = int(len(truth))
    correct = int(np.count_nonzero(truth == pred)) if total else 0
    by_char: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for sample, y, p in zip(eval_samples, truth, pred):
        record = by_char[sample.char]
        record[0] += 1
        if int(y) == int(p):
            record[1] += 1

    class_metrics = []
    for char, (support, class_correct) in sorted(by_char.items()):
        class_metrics.append(
            ClassMetric(
                char=char,
                support=support,
                correct=class_correct,
                accuracy=round(class_correct / support, 6) if support else math.nan,
                train_count=train_counts.get(char, 0),
            )
        )

    overall = {
        "samples": total,
        "correct": correct,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "classes_in_eval": len(class_metrics),
        "classes_in_train": len(idx_to_class),
    }
    return overall, class_metrics


def select_difficult(
    metrics: list[ClassMetric],
    threshold: float,
    top_k: int | None,
    target_count: int,
) -> list[dict[str, object]]:
    candidates = [m for m in metrics if m.support > 0]
    selected = [m for m in candidates if m.accuracy <= threshold]
    if top_k is not None and len(selected) < top_k:
        selected_chars = {m.char for m in selected}
        for metric in sorted(candidates, key=lambda x: (x.accuracy, -x.support, x.char)):
            if metric.char not in selected_chars:
                selected.append(metric)
                selected_chars.add(metric.char)
            if len(selected) >= top_k:
                break
    return [
        {
            "char": metric.char,
            "count": metric.train_count,
            "need_to_min_count": max(0, target_count - metric.train_count),
            "accuracy": metric.accuracy,
            "support": metric.support,
        }
        for metric in selected
        if metric.train_count < target_count
    ]


def run_training(args: argparse.Namespace) -> dict[str, object]:
    real_train = load_split_samples(args.data_root, args.train_csv, "real_train")
    augmented = []
    for manifest in args.aug_manifest or []:
        augmented.extend(load_manifest_samples(manifest))
    train_samples = real_train + augmented
    eval_csv = {"val": args.val_csv, "test": args.test_csv}[args.eval_split]
    eval_samples = load_split_samples(args.data_root, eval_csv, args.eval_split)

    classes = sorted({sample.char for sample in real_train})
    class_to_idx = {char: idx for idx, char in enumerate(classes)}
    train_x, train_y, train_kept = prepare_arrays(train_samples, class_to_idx, args.image_size)
    eval_x, eval_y, eval_kept = prepare_arrays(eval_samples, class_to_idx, args.image_size)
    if len(train_x) == 0:
        raise RuntimeError("No readable training images.")
    if len(eval_x) == 0:
        raise RuntimeError(f"No readable {args.eval_split} images.")

    if args.backend == "nearest_centroid":
        pred = predict_nearest_centroid(train_x, train_y, eval_x, len(classes))
    elif args.backend == "torch_cnn":
        pred = train_torch_cnn(
            train_x,
            train_y,
            eval_x,
            args.image_size,
            len(classes),
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.seed,
            args.device,
        )
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    train_counts = Counter(sample.char for sample in real_train)
    overall, class_metrics = compute_metrics(eval_kept, eval_y, pred, classes, train_counts)
    overall.update(
        {
            "backend": args.backend,
            "eval_split": args.eval_split,
            "real_train_samples": len(real_train),
            "augmented_train_samples": len(augmented),
            "readable_train_samples": len(train_kept),
            "readable_eval_samples": len(eval_kept),
            "image_size": args.image_size,
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    metric_rows = [asdict(metric) for metric in class_metrics]
    write_dict_rows(args.out_dir / "per_class_metrics.csv", metric_rows, ["char", "support", "correct", "accuracy", "train_count"])

    difficult = select_difficult(class_metrics, args.difficulty_threshold, args.difficulty_top_k, args.target_count)
    write_dict_rows(
        args.out_dir / "difficult_chars.csv",
        difficult,
        ["char", "count", "need_to_min_count", "accuracy", "support"],
    )
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    return overall


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a character classifier and export metrics.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--val_csv", type=Path, required=True)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--aug_manifest", action="append", type=Path)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["nearest_centroid", "torch_cnn"], default="nearest_centroid")
    parser.add_argument("--eval_split", choices=["val", "test"], default="val")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--difficulty_threshold", type=float, default=0.6)
    parser.add_argument("--difficulty_top_k", type=int)
    parser.add_argument("--target_count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
