"""Local small-model and lightweight-feature clustering for emoji images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageSequence, UnidentifiedImageError
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from .config import CLUSTER_CSV_PATH, CLUSTER_FEATURES_PATH, IMAGE_EXTENSIONS, IMAGES_DIR


CLIP_LABELS = [
    ("real_person", "a photo or selfie of a real person"),
    ("real_people_group", "a photo of real people"),
    ("cartoon_character", "an anime cartoon character sticker"),
    ("drawn_comic", "a drawn comic or illustration sticker"),
    ("text_meme", "a meme image dominated by text"),
    ("screenshot", "a screenshot of a chat or app interface"),
    ("animal", "an animal sticker"),
    ("object_food", "an object or food sticker"),
]

DEFAULT_CLIP_DISTANCE_THRESHOLD = 0.18


CSV_COLUMNS = [
    "path",
    "file_name",
    "asset_type",
    "asset_confidence",
    "visual_cluster_id",
    "near_duplicate_group",
    "cluster_size",
    "duplicate_group_size",
    "width",
    "height",
    "aspect_ratio",
    "suffix",
    "dhash",
    "ahash",
    "color_diversity",
    "edge_density",
    "saturation_mean",
    "brightness_mean",
    "encoder",
]


class DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def iter_images(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def relative_image_path(path: Path, images_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(images_dir.resolve().parent))
    except ValueError:
        return str(Path(images_dir.name) / path.name)


def open_image(path: Path) -> Image.Image:
    image = Image.open(path)
    try:
        image.seek(0)
    except EOFError:
        pass
    if getattr(image, "is_animated", False):
        image = next(ImageSequence.Iterator(image))
    return image.convert("RGB")


def bits_to_hex(bits: np.ndarray) -> str:
    value = 0
    for bit in bits.astype(bool).flatten().tolist():
        value = (value << 1) | int(bit)
    width = math.ceil(bits.size / 4)
    return f"{value:0{width}x}"


def average_hash(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.float32)
    return bits_to_hex(values > values.mean())


def difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.float32)
    return bits_to_hex(values[:, 1:] > values[:, :-1])


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def handcrafted_features(image: Image.Image) -> tuple[np.ndarray, dict[str, float]]:
    resized = image.resize((64, 64), Image.Resampling.LANCZOS)
    rgb = np.asarray(resized, dtype=np.float32) / 255.0

    hist_parts = []
    for channel in range(3):
        hist, _ = np.histogram(rgb[:, :, channel], bins=8, range=(0.0, 1.0), density=False)
        hist_parts.append(hist.astype(np.float32) / rgb[:, :, channel].size)
    hist_vec = np.concatenate(hist_parts)

    maxc = rgb.max(axis=2)
    minc = rgb.min(axis=2)
    saturation = np.where(maxc > 1e-6, (maxc - minc) / np.maximum(maxc, 1e-6), 0.0)
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    grad_y, grad_x = np.gradient(luminance)
    edge = np.sqrt(grad_x**2 + grad_y**2)

    small = image.resize((32, 32), Image.Resampling.BILINEAR)
    quantized = (np.asarray(small, dtype=np.uint8) // 32).reshape(-1, 3)
    color_diversity = len({tuple(row.tolist()) for row in quantized}) / len(quantized)

    width, height = image.size
    scalars = np.asarray(
        [
            luminance.mean(),
            luminance.std(),
            saturation.mean(),
            saturation.std(),
            edge.mean(),
            edge.std(),
            color_diversity,
            min(width / max(height, 1), 5.0) / 5.0,
            math.log1p(width) / 10.0,
            math.log1p(height) / 10.0,
        ],
        dtype=np.float32,
    )
    stats = {
        "color_diversity": float(color_diversity),
        "edge_density": float(edge.mean()),
        "saturation_mean": float(saturation.mean()),
        "brightness_mean": float(luminance.mean()),
    }
    return np.concatenate([hist_vec, scalars]).astype(np.float32), stats


def heuristic_asset_type(stats: dict[str, float], suffix: str) -> tuple[str, float]:
    if suffix.lower() == ".gif":
        return "animated", 0.8
    color_diversity = stats["color_diversity"]
    edge_density = stats["edge_density"]
    saturation = stats["saturation_mean"]
    if color_diversity >= 0.56 and edge_density <= 0.105:
        return "photo_like", 0.55
    if saturation >= 0.28 and edge_density >= 0.08:
        return "cartoon_or_sticker", 0.5
    if color_diversity <= 0.22:
        return "text_or_flat_graphic", 0.45
    return "uncertain", 0.25


def encode_clip(paths: list[Path], model_name: str, batch_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        images = [open_image(path) for path in paths[start : start + batch_size]]
        batch = model.encode(
            images,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings.append(np.asarray(batch, dtype=np.float32))
    image_embeddings = np.vstack(embeddings) if embeddings else np.empty((0, 0), dtype=np.float32)
    text_embeddings = model.encode(
        [label for _, label in CLIP_LABELS],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    scores = image_embeddings @ text_embeddings.T
    best = scores.argmax(axis=1) if len(scores) else np.asarray([], dtype=int)
    second = np.partition(scores, -2, axis=1)[:, -2] if scores.shape[1] >= 2 else np.zeros(len(scores))
    labels = []
    for index, best_index in enumerate(best.tolist()):
        labels.append(
            {
                "asset_type": CLIP_LABELS[best_index][0],
                "asset_confidence": float(scores[index, best_index] - second[index]),
                "clip_scores": {
                    CLIP_LABELS[label_index][0]: float(scores[index, label_index])
                    for label_index in range(len(CLIP_LABELS))
                },
            }
        )
    return image_embeddings, {"labels": labels, "model": model_name}


def build_features(
    *,
    images_dir: Path,
    output_path: Path,
    encoder: str,
    clip_model: str,
    batch_size: int,
    limit: int,
) -> dict[str, Any]:
    paths = iter_images(images_dir)
    if limit:
        paths = paths[:limit]
    records: list[dict[str, Any]] = []
    hand_features: list[np.ndarray] = []

    for path in paths:
        try:
            image = open_image(path)
        except (OSError, UnidentifiedImageError) as exc:
            records.append(
                {
                    "path": relative_image_path(path, images_dir),
                    "file_name": path.name,
                    "error": str(exc),
                }
            )
            continue
        feature, stats = handcrafted_features(image)
        asset_type, confidence = heuristic_asset_type(stats, path.suffix)
        width, height = image.size
        records.append(
            {
                "path": relative_image_path(path, images_dir),
                "file_name": path.name,
                "absolute_path": str(path.resolve()),
                "width": width,
                "height": height,
                "aspect_ratio": width / max(height, 1),
                "suffix": path.suffix.lower(),
                "dhash": difference_hash(image),
                "ahash": average_hash(image),
                "asset_type": asset_type,
                "asset_confidence": confidence,
                **stats,
            }
        )
        hand_features.append(feature)

    valid_indices = [index for index, record in enumerate(records) if "error" not in record]
    hand_matrix = np.vstack(hand_features) if hand_features else np.empty((0, 0), dtype=np.float32)
    feature_matrix = hand_matrix
    clip_meta: dict[str, Any] = {}
    encoder_used = "handcrafted"
    if encoder == "clip":
        valid_paths = [Path(records[index]["absolute_path"]) for index in valid_indices]
        clip_matrix, clip_meta = encode_clip(valid_paths, clip_model, batch_size)
        feature_matrix = clip_matrix
        encoder_used = "clip"
        for record_index, label in zip(valid_indices, clip_meta["labels"], strict=False):
            records[record_index].update(label)

    payload = {
        "version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "images_dir": str(images_dir),
        "encoder": encoder_used,
        "clip_model": clip_model if encoder_used == "clip" else "",
        "records": records,
        "valid_indices": valid_indices,
        "features": feature_matrix,
        "handcrafted_features": hand_matrix,
        "clip_meta": clip_meta,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "output": str(output_path),
        "encoder": encoder_used,
        "records": len(records),
        "valid_records": len(valid_indices),
        "feature_dim": int(feature_matrix.shape[1]) if feature_matrix.size else 0,
    }


def load_features(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def cluster_features(payload: dict[str, Any], *, distance_threshold: float | None) -> dict[str, Any]:
    records = payload["records"]
    valid_indices = payload["valid_indices"]
    features = np.asarray(payload["features"], dtype=np.float32)
    if not len(valid_indices):
        return {"records": records, "n_clusters": 0, "duplicate_groups": 0}

    encoder = payload.get("encoder", "handcrafted")
    if len(valid_indices) == 1:
        labels = np.asarray([0], dtype=int)
    elif encoder == "clip":
        threshold = DEFAULT_CLIP_DISTANCE_THRESHOLD if distance_threshold is None else distance_threshold
        labels = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=threshold,
        ).fit_predict(features)
    else:
        threshold = 2.4 if distance_threshold is None else distance_threshold
        scaled = StandardScaler().fit_transform(features)
        labels = AgglomerativeClustering(
            n_clusters=None,
            metric="euclidean",
            linkage="ward",
            distance_threshold=threshold,
        ).fit_predict(scaled)

    cluster_counts = Counter(labels.tolist())
    for record_index, label in zip(valid_indices, labels.tolist(), strict=False):
        records[record_index]["visual_cluster_id"] = f"visual_{label:04d}"
        records[record_index]["cluster_size"] = cluster_counts[label]

    hashes = [records[index]["dhash"] for index in valid_indices]
    dsu = DisjointSet(len(valid_indices))
    for left in range(len(hashes)):
        for right in range(left + 1, len(hashes)):
            if hamming_hex(hashes[left], hashes[right]) <= 4:
                dsu.union(left, right)
    roots = [dsu.find(index) for index in range(len(valid_indices))]
    root_to_id = {root: group_id for group_id, root in enumerate(sorted(set(roots)))}
    group_counts = Counter(root_to_id[root] for root in roots)
    for local_index, record_index in enumerate(valid_indices):
        group_id = root_to_id[roots[local_index]]
        records[record_index]["near_duplicate_group"] = f"dup_{group_id:04d}"
        records[record_index]["duplicate_group_size"] = group_counts[group_id]

    return {
        "records": records,
        "n_clusters": len(cluster_counts),
        "duplicate_groups": len(group_counts),
    }


def export_cluster_csv(records: list[dict[str, Any]], output_path: Path, encoder: str) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path(".") else None
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        rows = 0
        for record in records:
            if "error" in record:
                continue
            row = {column: record.get(column, "") for column in CSV_COLUMNS}
            row["encoder"] = encoder
            writer.writerow(row)
            rows += 1
    return {"output": str(output_path), "rows": rows}


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    feature_summary = build_features(
        images_dir=Path(args.images_dir).expanduser(),
        output_path=Path(args.features).expanduser(),
        encoder=args.encoder,
        clip_model=args.clip_model,
        batch_size=max(1, args.batch_size),
        limit=max(0, args.limit),
    )
    payload = load_features(Path(args.features).expanduser())
    clustered = cluster_features(payload, distance_threshold=args.distance_threshold)
    csv_summary = export_cluster_csv(
        clustered["records"],
        Path(args.output).expanduser(),
        payload.get("encoder", args.encoder),
    )
    return {**feature_summary, **clustered, **csv_summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster emoji images with local lightweight visual features.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(run: argparse.ArgumentParser) -> None:
        run.add_argument("--images-dir", default=str(IMAGES_DIR), help="Local image corpus directory.")
        run.add_argument("--features", default=str(CLUSTER_FEATURES_PATH), help="Feature pickle output/input path.")
        run.add_argument("--output", default=str(CLUSTER_CSV_PATH), help="Cluster CSV output path.")
        run.add_argument("--encoder", choices=("handcrafted", "clip"), default="handcrafted", help="Feature encoder.")
        run.add_argument("--clip-model", default="clip-ViT-B-32", help="SentenceTransformer CLIP model.")
        run.add_argument("--batch-size", type=int, default=32, help="Embedding batch size for CLIP.")
        run.add_argument("--limit", type=int, default=0, help="Process at most N images. 0 means no limit.")
        run.add_argument(
            "--distance-threshold",
            type=float,
            default=None,
            help="Override clustering threshold. Lower values create finer clusters.",
        )

    run = subparsers.add_parser("run", help="Build features, cluster, and export CSV.")
    add_common(run)

    features = subparsers.add_parser("features", help="Build and save local image features.")
    add_common(features)

    cluster = subparsers.add_parser("cluster", help="Cluster saved features and export CSV.")
    cluster.add_argument("--features", default=str(CLUSTER_FEATURES_PATH), help="Feature pickle input path.")
    cluster.add_argument("--output", default=str(CLUSTER_CSV_PATH), help="Cluster CSV output path.")
    cluster.add_argument("--distance-threshold", type=float, default=None, help="Override clustering threshold.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "run":
        summary = run_pipeline(args)
        summary.pop("records", None)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "features":
        summary = build_features(
            images_dir=Path(args.images_dir).expanduser(),
            output_path=Path(args.features).expanduser(),
            encoder=args.encoder,
            clip_model=args.clip_model,
            batch_size=max(1, args.batch_size),
            limit=max(0, args.limit),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "cluster":
        payload = load_features(Path(args.features).expanduser())
        clustered = cluster_features(payload, distance_threshold=args.distance_threshold)
        csv_summary = export_cluster_csv(
            clustered["records"],
            Path(args.output).expanduser(),
            payload.get("encoder", "handcrafted"),
        )
        summary = {**clustered, **csv_summary}
        summary.pop("records", None)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
