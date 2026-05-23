"""Shared caption schema used by indexing and external annotation workflows."""

from __future__ import annotations

import json
from typing import Any


FIELD_NAMES = [
    "image_composition",
    "character_name",
    "expression",
    "action",
    "subjective_emotion",
    "text_in_image",
    "usage_context",
    "wechat_keyword",
    "manual_tags",
    "notes",
]

FIELD_LABELS = {
    "image_composition": "图片构成",
    "character_name": "角色名称",
    "expression": "表情",
    "action": "动作",
    "subjective_emotion": "主观情绪",
    "text_in_image": "图中文字",
    "usage_context": "适用聊天场景",
    "wechat_keyword": "微信含义词候选",
    "manual_tags": "人工标签",
    "notes": "补充",
}

PLACEHOLDERS = {"", "none", "NONE", "无", "未知", "null", "NULL", "N/A", "n/a"}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        return "；".join(
            text
            for item in value
            if (text := normalize_text(item))
        )
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def is_placeholder(value: str) -> bool:
    return value.strip() in PLACEHOLDERS


def caption_template(fill: str = "") -> dict[str, str]:
    return {field: fill for field in FIELD_NAMES}


def normalize_caption(caption: dict[str, Any], *, fill_missing: str = "NONE") -> dict[str, str]:
    normalized: dict[str, str] = {}
    for field in FIELD_NAMES:
        value = normalize_text(caption.get(field))
        normalized[field] = value if value else fill_missing
    return normalized


def semantic_text(caption: dict[str, Any], fallback: str = "") -> str:
    text = "；".join(
        normalize_text(caption.get(field))
        for field in FIELD_NAMES
        if normalize_text(caption.get(field)) and not is_placeholder(normalize_text(caption.get(field)))
    )
    return text or fallback.strip()


def parse_caption_json(raw_content: str) -> tuple[dict[str, Any], str]:
    try:
        caption = json.loads(raw_content)
    except json.JSONDecodeError:
        caption = {"raw": raw_content.strip()}
    if not isinstance(caption, dict):
        caption = {"raw": normalize_text(caption)}
    normalized = normalize_caption(caption)
    return normalized, semantic_text(normalized, fallback=raw_content)


def caption_index_item(
    *,
    path: str,
    caption: dict[str, Any],
    model: str,
    endpoint: str,
    source: str,
) -> dict[str, Any]:
    normalized = normalize_caption(caption)
    return {
        "path": path,
        "ok": True,
        "model": model,
        "endpoint": endpoint,
        "source": source,
        "caption": normalized,
        "semantic_text": semantic_text(normalized),
    }
