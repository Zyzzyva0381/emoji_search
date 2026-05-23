"""Central paths and environment defaults for the emoji search project."""

from __future__ import annotations

import os
from pathlib import Path

from .envfile import load_env_file


PROJ_ROOT = Path(__file__).resolve().parents[1]
load_env_file(PROJ_ROOT / ".env")

DEFAULT_VECTOR_MODEL = "BAAI/bge-m3"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def path_from_env(name: str, default: Path) -> Path:
    path = Path(os.getenv(name, str(default))).expanduser()
    return path if path.is_absolute() else PROJ_ROOT / path


IMAGES_DIR = path_from_env("EMOJI_IMAGES_DIR", PROJ_ROOT / "images")
CAPTION_INDEX_PATH = path_from_env("EMOJI_CAPTION_INDEX", PROJ_ROOT / "image_index.jsonl")
VECTOR_INDEX_PATH = path_from_env("EMOJI_VECTOR_INDEX", PROJ_ROOT / "image_vectors.pkl")
MANIFEST_PATH = path_from_env("EMOJI_IMAGE_MANIFEST", PROJ_ROOT / "image_manifest.jsonl")
ANNOTATION_BATCHES_DIR = path_from_env("EMOJI_ANNOTATION_BATCHES_DIR", PROJ_ROOT / "annotation_batches")
CLUSTER_FEATURES_PATH = path_from_env("EMOJI_CLUSTER_FEATURES", PROJ_ROOT / "image_cluster_features.pkl")
CLUSTER_CSV_PATH = path_from_env("EMOJI_CLUSTER_CSV", PROJ_ROOT / "image_clusters.csv")
VECTOR_MODEL = os.getenv("EMOJI_VECTOR_MODEL", DEFAULT_VECTOR_MODEL)
