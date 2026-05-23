#!/usr/bin/env python3
"""FastAPI backend for emoji image management and semantic search."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from emoji_search.config import CAPTION_INDEX_PATH, IMAGE_EXTENSIONS, IMAGES_DIR, PROJ_ROOT as ROOT, VECTOR_INDEX_PATH, VECTOR_MODEL
from emoji_search.envfile import api_settings_from_env
from main import describe_image
from semantic_search import (
    FIELD_NAMES,
    load_index,
    loaded_models,
    remove_paths_from_index,
    save_index,
    search_payload,
    unload_models,
    upsert_index_record,
    warm_model,
)


INDEX_LOCK = threading.RLock()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    fields: list[str] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=100)
    score_mode: Literal["max", "mean", "sum"] = "max"


class ImageRecord(BaseModel):
    path: str
    url: str
    indexed: bool
    fields: dict[str, str] = Field(default_factory=dict)


class SearchResult(BaseModel):
    path: str
    url: str
    score: float
    best_field: str
    field_scores: dict[str, float]
    fields: dict[str, str]


class CaptionStatus(BaseModel):
    total_images: int
    captioned_images: int
    vector_indexed_images: int
    vector_index_exists: bool
    loaded_models: list[str]


app = FastAPI(title="Emoji Semantic Search")
app.mount("/app", StaticFiles(directory=ROOT / "static", html=True), name="app")


def ensure_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def image_url(path: str) -> str:
    return f"/images/{Path(path).name}"


def relative_image_path(path: Path) -> str:
    return str(Path(IMAGES_DIR.name) / path.name)


def safe_image_path(path: str) -> Path:
    name = Path(path).name
    candidate = (IMAGES_DIR / name).resolve()
    if IMAGES_DIR.resolve() not in candidate.parents and candidate != IMAGES_DIR.resolve():
        raise HTTPException(status_code=400, detail="Invalid image path")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return candidate


def iter_image_files() -> list[Path]:
    ensure_dirs()
    return sorted(
        path
        for path in IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_caption_items() -> list[dict[str, Any]]:
    if not CAPTION_INDEX_PATH.exists():
        return []
    items: list[dict[str, Any]] = []
    with CAPTION_INDEX_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def save_caption_items(items: list[dict[str, Any]]) -> None:
    CAPTION_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CAPTION_INDEX_PATH.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def caption_by_path() -> dict[str, dict[str, Any]]:
    items = load_caption_items()
    return {
        item["path"]: item
        for item in items
        if item.get("ok") and item.get("path")
    }


def remove_caption_paths(paths: set[str]) -> int:
    items = load_caption_items()
    kept = [item for item in items if item.get("path") not in paths]
    save_caption_items(kept)
    return len(items) - len(kept)


def upsert_caption_item(item: dict[str, Any]) -> None:
    items = [existing for existing in load_caption_items() if existing.get("path") != item.get("path")]
    items.append(item)
    save_caption_items(items)


def current_vector_payload() -> dict[str, Any] | None:
    if not VECTOR_INDEX_PATH.exists():
        return None
    return load_index(VECTOR_INDEX_PATH)


def vector_paths() -> set[str]:
    payload = current_vector_payload()
    return set(payload["paths"]) if payload else set()


def active_vector_model() -> str:
    payload = current_vector_payload()
    if payload and payload.get("model"):
        return str(payload["model"])
    return VECTOR_MODEL


def append_or_update_vector(path: str, caption: dict[str, str]) -> None:
    upsert_index_record(
        VECTOR_INDEX_PATH,
        path=path,
        record_fields=caption,
        model_name=VECTOR_MODEL,
        fields=FIELD_NAMES,
    )


def rebuild_missing_vectors() -> int:
    captions = caption_by_path()
    existing = vector_paths()
    missing = [path for path in captions if path not in existing and (ROOT / path).exists()]
    for path in missing:
        append_or_update_vector(path, captions[path].get("caption") or {})
    return len(missing)


def remove_stale_vectors() -> int:
    existing_images = {relative_image_path(path) for path in iter_image_files()}
    indexed = vector_paths()
    stale = sorted(indexed - existing_images)
    if stale:
        return remove_paths_from_index(VECTOR_INDEX_PATH, stale)
    return 0


def caption_image(path: Path) -> dict[str, Any]:
    settings = api_settings_from_env(ROOT / ".env")
    if not settings["api_key"]:
        raise HTTPException(status_code=503, detail="Missing OPENAI_API_KEY or API_KEY for automatic image captioning")
    item = describe_image(
        path=path,
        relative_path=relative_image_path(path),
        base_url=settings["base_url"],
        api_key=settings["api_key"],
        model=settings["model"],
        detail=os.getenv("VISION_DETAIL", "low"),
        max_tokens=int(os.getenv("VISION_MAX_TOKENS", "500")),
        use_response_format=os.getenv("VISION_NO_RESPONSE_FORMAT", "0") != "1",
        timeout=int(os.getenv("VISION_TIMEOUT", "120")),
        retries=int(os.getenv("VISION_RETRIES", "1")),
    )
    if not item.get("ok"):
        raise HTTPException(status_code=502, detail=item.get("error") or "Image captioning failed")
    upsert_caption_item(item)
    return item


def index_existing_image(path: Path, force_caption: bool = False) -> ImageRecord:
    rel = relative_image_path(path)
    captions = caption_by_path()
    item = captions.get(rel)
    if force_caption or item is None:
        item = caption_image(path)
    caption = item.get("caption") or {}
    append_or_update_vector(rel, caption)
    return ImageRecord(
        path=rel,
        url=image_url(rel),
        indexed=True,
        fields={field: str(caption.get(field, "")) for field in FIELD_NAMES},
    )


def copy_file_to_images(upload: UploadFile) -> tuple[Path, str]:
    ensure_dirs()
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {suffix or 'unknown'}")
    digest = hashlib.sha256()
    tmp_path = IMAGES_DIR / f".upload-{time.time_ns()}{suffix}"
    with tmp_path.open("wb") as f:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            f.write(chunk)
    final_path = IMAGES_DIR / f"{int(time.time())}_{digest.hexdigest()[:16]}{suffix}"
    if final_path.exists():
        tmp_path.unlink(missing_ok=True)
    else:
        shutil.move(str(tmp_path), final_path)
    return final_path, relative_image_path(final_path)


@app.get("/")
def root() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/status", response_model=CaptionStatus)
def status() -> CaptionStatus:
    images = iter_image_files()
    captions = caption_by_path()
    vectors = vector_paths()
    return CaptionStatus(
        total_images=len(images),
        captioned_images=len(captions),
        vector_indexed_images=len(vectors),
        vector_index_exists=VECTOR_INDEX_PATH.exists(),
        loaded_models=loaded_models(),
    )


@app.get("/api/fields")
def fields() -> dict[str, list[str]]:
    return {"fields": FIELD_NAMES}


@app.get("/api/images", response_model=list[ImageRecord])
def images() -> list[ImageRecord]:
    captions = caption_by_path()
    vectors = vector_paths()
    records: list[ImageRecord] = []
    for path in iter_image_files():
        rel = relative_image_path(path)
        caption = captions.get(rel, {}).get("caption") or {}
        records.append(
            ImageRecord(
                path=rel,
                url=image_url(rel),
                indexed=rel in vectors,
                fields={field: str(caption.get(field, "")) for field in FIELD_NAMES},
            )
        )
    return records


@app.get("/images/{filename}")
def serve_image(filename: str) -> FileResponse:
    path = safe_image_path(filename)
    return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0])


@app.post("/api/images", response_model=ImageRecord)
def upload_image(file: UploadFile = File(...), auto_index: bool = True) -> ImageRecord:
    with INDEX_LOCK:
        saved_path, rel = copy_file_to_images(file)
        fields: dict[str, str] = {}
        indexed = False
        if auto_index:
            item = caption_image(saved_path)
            fields = {field: str((item.get("caption") or {}).get(field, "")) for field in FIELD_NAMES}
            append_or_update_vector(rel, item.get("caption") or {})
            indexed = True
        return ImageRecord(path=rel, url=image_url(rel), indexed=indexed, fields=fields)


@app.delete("/api/images/{filename}")
def delete_image(filename: str) -> dict[str, Any]:
    with INDEX_LOCK:
        path = safe_image_path(filename)
        rel = relative_image_path(path)
        path.unlink()
        removed_captions = remove_caption_paths({rel})
        removed_vectors = remove_paths_from_index(VECTOR_INDEX_PATH, [rel])
        return {"deleted": rel, "removed_captions": removed_captions, "removed_vectors": removed_vectors}


@app.post("/api/index/sync")
def sync_index() -> dict[str, int]:
    with INDEX_LOCK:
        removed = remove_stale_vectors()
        added = rebuild_missing_vectors()
        return {"added_vectors": added, "removed_stale_vectors": removed}


@app.delete("/api/index/{filename}")
def delete_vector_index(filename: str) -> dict[str, Any]:
    with INDEX_LOCK:
        rel = relative_image_path(safe_image_path(filename))
        removed = remove_paths_from_index(VECTOR_INDEX_PATH, [rel])
        return {"path": rel, "removed_vectors": removed}


@app.post("/api/index/{filename}", response_model=ImageRecord)
def index_image(filename: str, force_caption: bool = False) -> ImageRecord:
    with INDEX_LOCK:
        path = safe_image_path(filename)
        return index_existing_image(path, force_caption=force_caption)


@app.delete("/api/index")
def delete_all_vectors() -> dict[str, Any]:
    with INDEX_LOCK:
        existed = VECTOR_INDEX_PATH.exists()
        VECTOR_INDEX_PATH.unlink(missing_ok=True)
        save_index(
            {
                "version": 1,
                "model": VECTOR_MODEL,
                "fields": FIELD_NAMES,
                "paths": [],
                "texts": [],
                "embeddings": {},
                "valid_masks": {},
            },
            VECTOR_INDEX_PATH,
        )
        return {"deleted": existed, "path": str(VECTOR_INDEX_PATH)}


@app.post("/api/model/load")
def load_model() -> dict[str, Any]:
    model_name = active_vector_model()
    started_at = time.time()
    warm_model(model_name)
    return {
        "model": model_name,
        "loaded_models": loaded_models(),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }


@app.post("/api/model/unload")
def unload_model() -> dict[str, Any]:
    unload_models()
    return {"loaded_models": loaded_models()}


@app.post("/api/search", response_model=list[SearchResult])
def search(request: SearchRequest) -> list[dict[str, Any]]:
    if not VECTOR_INDEX_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Search index not found: {VECTOR_INDEX_PATH}")
    try:
        payload = load_index(VECTOR_INDEX_PATH)
        results = search_payload(
            payload=payload,
            query=request.query,
            selected_fields=request.fields,
            top_k=request.top_k,
            score_mode=request.score_mode,
        )
        for result in results:
            result["url"] = image_url(result["path"])
        return results
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
