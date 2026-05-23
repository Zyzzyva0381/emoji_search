#!/usr/bin/env python3
"""Field-level semantic search over captioned emoji images.

Build:
    uv run python semantic_search.py build --input image_index.jsonl --output image_vectors.pkl

Search:
    uv run python semantic_search.py search --index image_vectors.pkl --query "委屈哭哭" --fields expression subjective_emotion --top-k 10
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from emoji_search.caption_schema import FIELD_NAMES, PLACEHOLDERS, is_placeholder, normalize_text


DEFAULT_MODEL = "BAAI/bge-m3"
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


@dataclass
class CaptionRecord:
    path: str
    fields: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and query a sentence-transformers emoji search index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Encode image caption fields and serialize a vector index.")
    build.add_argument("--input", default="image_index.jsonl", help="Caption JSONL produced by main.py.")
    build.add_argument("--output", default="image_vectors.pkl", help="Serialized vector index file.")
    build.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name or local path.")
    build.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    build.add_argument(
        "--fields",
        nargs="*",
        default=FIELD_NAMES,
        help="Caption fields to index. Defaults to all known caption fields.",
    )

    search = subparsers.add_parser("search", help="Search image paths by a short text query.")
    search.add_argument("--index", default="image_vectors.pkl", help="Serialized vector index file.")
    search.add_argument("--query", required=True, help="User query text.")
    search.add_argument(
        "--fields",
        nargs="*",
        default=[],
        help="0 or more fields to match. Empty means all indexed fields.",
    )
    search.add_argument("--top-k", type=int, default=10, help="Number of image paths to return.")
    search.add_argument(
        "--score-mode",
        choices=("max", "mean", "sum"),
        default="max",
        help="How to combine scores across selected fields.",
    )
    search.add_argument(
        "--json",
        action="store_true",
        help="Print JSON result objects instead of plain paths.",
    )
    return parser.parse_args()


def load_caption_records(path: Path, fields: list[str]) -> list[CaptionRecord]:
    records: list[CaptionRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

            if not item.get("ok") or not item.get("path"):
                continue

            caption = item.get("caption") or {}
            record_fields = {field: normalize_text(caption.get(field)) for field in fields}
            records.append(CaptionRecord(path=item["path"], fields=record_fields))
    return records


def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def get_model(model_name: str) -> SentenceTransformer:
    model = _MODEL_CACHE.get(model_name)
    if model is None:
        try:
            model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            model = SentenceTransformer(model_name)
        _MODEL_CACHE[model_name] = model
    return model


def unload_models() -> None:
    _MODEL_CACHE.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def loaded_models() -> list[str]:
    return sorted(_MODEL_CACHE)


def warm_model(model_name: str) -> None:
    model = get_model(model_name)
    model.encode(
        ["warmup"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def build_index(input_path: Path, output_path: Path, model_name: str, fields: list[str], batch_size: int) -> None:
    records = load_caption_records(input_path, fields)
    if not records:
        raise ValueError(f"No successful caption records found in {input_path}")

    print(f"Loading model: {model_name}")
    model = get_model(model_name)
    embeddings: dict[str, np.ndarray] = {}
    valid_masks: dict[str, np.ndarray] = {}

    for field in fields:
        valid_masks[field] = np.asarray(
            [not is_placeholder(record.fields[field]) for record in records],
            dtype=bool,
        )
        texts = [
            record.fields[field] if not is_placeholder(record.fields[field]) else ""
            for record in records
        ]
        print(f"Encoding field: {field} ({len(texts)} text(s))")
        embeddings[field] = encode_texts(model, texts, batch_size)

    payload = {
        "version": 1,
        "model": model_name,
        "fields": fields,
        "paths": [record.path for record in records],
        "texts": [record.fields for record in records],
        "embeddings": embeddings,
        "valid_masks": valid_masks,
    }
    save_index(payload, output_path)
    print(f"Saved {len(records)} record(s) to {output_path}")


def save_index(payload: dict[str, Any], index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True) if index_path.parent != Path(".") else None
    with index_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(index_path: Path) -> dict[str, Any]:
    with index_path.open("rb") as f:
        payload = pickle.load(f)
    required = {"version", "model", "fields", "paths", "texts", "embeddings"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Invalid index, missing key(s): {', '.join(sorted(missing))}")
    if "valid_masks" not in payload:
        payload["valid_masks"] = {
            field: np.ones(len(payload["paths"]), dtype=bool)
            for field in payload["fields"]
        }
    return payload


def empty_index(model_name: str = DEFAULT_MODEL, fields: list[str] | None = None) -> dict[str, Any]:
    return {
        "version": 1,
        "model": model_name,
        "fields": list(fields or FIELD_NAMES),
        "paths": [],
        "texts": [],
        "embeddings": {},
        "valid_masks": {},
    }


def encode_record(
    model_name: str,
    fields: list[str],
    record_fields: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    model = get_model(model_name)
    embeddings: dict[str, np.ndarray] = {}
    valid_masks: dict[str, np.ndarray] = {}
    for field in fields:
        text = normalize_text(record_fields.get(field))
        valid = not is_placeholder(text)
        embeddings[field] = encode_texts(model, [text if valid else ""], batch_size=1)
        valid_masks[field] = np.asarray([valid], dtype=bool)
    return embeddings, valid_masks


def remove_paths_from_payload(payload: dict[str, Any], paths: set[str]) -> dict[str, Any]:
    if not paths or not payload["paths"]:
        return payload

    keep_indices = [index for index, path in enumerate(payload["paths"]) if path not in paths]
    payload["paths"] = [payload["paths"][index] for index in keep_indices]
    payload["texts"] = [payload["texts"][index] for index in keep_indices]
    for field in payload["fields"]:
        if field in payload["embeddings"]:
            payload["embeddings"][field] = payload["embeddings"][field][keep_indices]
        if field in payload["valid_masks"]:
            payload["valid_masks"][field] = payload["valid_masks"][field][keep_indices]
    return payload


def remove_paths_from_index(index_path: Path, paths: list[str]) -> int:
    if not index_path.exists():
        return 0
    payload = load_index(index_path)
    before = len(payload["paths"])
    remove_paths_from_payload(payload, set(paths))
    save_index(payload, index_path)
    return before - len(payload["paths"])


def upsert_index_record(
    index_path: Path,
    path: str,
    record_fields: dict[str, str],
    model_name: str = DEFAULT_MODEL,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    payload = load_index(index_path) if index_path.exists() else empty_index(model_name, fields)
    payload = remove_paths_from_payload(payload, {path})
    index_fields = payload["fields"]
    embeddings, valid_masks = encode_record(payload["model"], index_fields, record_fields)

    payload["paths"].append(path)
    payload["texts"].append({field: normalize_text(record_fields.get(field)) for field in index_fields})
    for field in index_fields:
        if field not in payload["embeddings"] or len(payload["embeddings"][field]) == 0:
            payload["embeddings"][field] = embeddings[field]
            payload["valid_masks"][field] = valid_masks[field]
        else:
            payload["embeddings"][field] = np.concatenate([payload["embeddings"][field], embeddings[field]], axis=0)
            payload["valid_masks"][field] = np.concatenate([payload["valid_masks"][field], valid_masks[field]], axis=0)

    save_index(payload, index_path)
    return payload


def validate_selected_fields(index_fields: list[str], selected_fields: list[str]) -> list[str]:
    if not selected_fields:
        return index_fields
    unknown = [field for field in selected_fields if field not in index_fields]
    if unknown:
        allowed = ", ".join(index_fields)
        raise ValueError(f"Unknown field(s): {', '.join(unknown)}. Allowed fields: {allowed}")
    return selected_fields


def combine_scores(
    field_scores: dict[str, np.ndarray],
    valid_masks: dict[str, np.ndarray],
    score_mode: str,
) -> np.ndarray:
    fields = list(field_scores)
    stacked = np.stack([field_scores[field] for field in fields], axis=1)
    mask = np.stack([valid_masks[field] for field in fields], axis=1)
    valid_count = mask.sum(axis=1)

    if score_mode == "max":
        masked = np.where(mask, stacked, -np.inf)
        return np.where(valid_count > 0, masked.max(axis=1), -1.0)
    if score_mode == "mean":
        summed = np.where(mask, stacked, 0.0).sum(axis=1)
        return np.where(valid_count > 0, summed / np.maximum(valid_count, 1), -1.0)
    if score_mode == "sum":
        summed = np.where(mask, stacked, 0.0).sum(axis=1)
        return np.where(valid_count > 0, summed, -1.0)
    raise ValueError(f"Unsupported score mode: {score_mode}")


def search_index(
    index_path: Path,
    query: str,
    selected_fields: list[str] | None = None,
    top_k: int = 10,
    score_mode: str = "max",
) -> list[dict[str, Any]]:
    payload = load_index(index_path)
    return search_payload(payload, query, selected_fields, top_k, score_mode)


def search_payload(
    payload: dict[str, Any],
    query: str,
    selected_fields: list[str] | None = None,
    top_k: int = 10,
    score_mode: str = "max",
) -> list[dict[str, Any]]:
    if not payload["paths"]:
        return []
    fields = validate_selected_fields(payload["fields"], selected_fields or [])

    model = get_model(payload["model"])
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype(np.float32)

    field_scores = {
        field: payload["embeddings"][field] @ query_embedding
        for field in fields
    }
    valid_masks = {field: payload["valid_masks"][field] for field in fields}
    scores = combine_scores(field_scores, valid_masks, score_mode)
    top_k = max(1, min(top_k, len(scores)))
    top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]

    results: list[dict[str, Any]] = []
    for index in top_indices.tolist():
        per_field = {
            field: float(field_scores[field][index])
            for field in fields
            if valid_masks[field][index]
        }
        best_field = max(per_field, key=per_field.get) if per_field else ""
        results.append(
            {
                "path": payload["paths"][index],
                "score": float(scores[index]),
                "best_field": best_field,
                "field_scores": per_field,
                "fields": payload["texts"][index],
            }
        )
    return results


def main() -> int:
    args = parse_args()
    if args.command == "build":
        build_index(
            input_path=Path(args.input),
            output_path=Path(args.output),
            model_name=args.model,
            fields=args.fields,
            batch_size=args.batch_size,
        )
        return 0

    if args.command == "search":
        results = search_index(
            index_path=Path(args.index),
            query=args.query,
            selected_fields=args.fields,
            top_k=args.top_k,
            score_mode=args.score_mode,
        )
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for result in results:
                print(result["path"])
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
