"""Export external annotation tasks and merge finished captions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .caption_schema import FIELD_LABELS, FIELD_NAMES, caption_index_item, caption_template, normalize_caption
from .config import ANNOTATION_BATCHES_DIR, CAPTION_INDEX_PATH, IMAGE_EXTENSIONS, IMAGES_DIR


PROMPT_TEMPLATE = """# 表情包多模态标注任务

请逐行读取本目录的 `batch_*.jsonl` 文件。每一行是一张待识别图片，`absolute_path` 是本机图片路径。

对每张图片输出一行 JSON 到 `captions.jsonl`，格式必须是：

```json
{{"path":"images/example.png","caption":{caption_example}}}
```

字段要求：
{field_lines}

规则：
- 字段值用简短中文短语，适合后续语义检索。
- 无法判断或不存在时写 `NONE`。
- `wechat_keyword` 用作微信表情含义词候选，尽量不超过 4 个汉字。
- `manual_tags` 是人工补充标签；自动标注时没有把握就写 `NONE`。
- 不要输出 Markdown，不要把多张图片写成一个 JSON 数组。
- `path` 必须原样复制任务里的 `path`。
"""


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


def load_captioned_paths(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    paths: set[str] = set()
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("ok") and item.get("path"):
                paths.add(str(item["path"]))
    return paths


def write_prompt(output_dir: Path) -> None:
    field_lines = "\n".join(f"- `{field}`: {FIELD_LABELS[field]}" for field in FIELD_NAMES)
    caption_example = json.dumps(caption_template("..."), ensure_ascii=False)
    (output_dir / "PROMPT.md").write_text(
        PROMPT_TEMPLATE.format(field_lines=field_lines, caption_example=caption_example),
        encoding="utf-8",
    )


def export_batches(
    *,
    images_dir: Path,
    output_dir: Path,
    caption_index: Path,
    batch_size: int,
    limit: int,
    include_captioned: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_prompt(output_dir)

    captioned = set() if include_captioned else load_captioned_paths(caption_index)
    tasks = []
    for path in iter_images(images_dir):
        rel = relative_image_path(path, images_dir)
        if rel in captioned:
            continue
        tasks.append(
            {
                "path": rel,
                "absolute_path": str(path.resolve()),
                "caption": caption_template(),
            }
        )
        if limit and len(tasks) >= limit:
            break

    batch_paths: list[str] = []
    for batch_index in range(0, len(tasks), batch_size):
        batch = tasks[batch_index : batch_index + batch_size]
        batch_path = output_dir / f"batch_{batch_index // batch_size:03d}.jsonl"
        with batch_path.open("w", encoding="utf-8") as f:
            for item in batch:
                f.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
                f.write("\n")
        batch_paths.append(str(batch_path))

    return {
        "output_dir": str(output_dir),
        "prompt": str(output_dir / "PROMPT.md"),
        "tasks": len(tasks),
        "batch_files": batch_paths,
        "skipped_captioned": 0 if include_captioned else len(captioned),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            item["_line_number"] = line_number
            items.append(item)
    return items


def coerce_external_item(item: dict[str, Any], *, source_model: str) -> dict[str, Any]:
    path = str(item.get("path") or "").strip()
    if not path:
        raise ValueError(f"line {item.get('_line_number', '?')}: missing path")
    raw_caption = item.get("caption")
    if not isinstance(raw_caption, dict):
        raw_caption = {field: item.get(field) for field in FIELD_NAMES}
    caption = normalize_caption(raw_caption)
    return caption_index_item(
        path=path,
        caption=caption,
        model=source_model,
        endpoint="external",
        source="external_annotation",
    )


def validate_external_captions(input_path: Path, *, source_model: str) -> dict[str, Any]:
    errors: list[str] = []
    paths: set[str] = set()
    valid = 0
    for item in read_jsonl(input_path):
        try:
            coerced = coerce_external_item(item, source_model=source_model)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if coerced["path"] in paths:
            errors.append(f"line {item.get('_line_number', '?')}: duplicate path {coerced['path']}")
            continue
        paths.add(coerced["path"])
        valid += 1
    return {"valid": valid, "errors": errors}


def merge_external_captions(
    *,
    input_path: Path,
    output_path: Path,
    source_model: str,
    overwrite: bool,
) -> dict[str, Any]:
    incoming = [coerce_external_item(item, source_model=source_model) for item in read_jsonl(input_path)]
    existing = []
    if output_path.exists():
        existing = read_jsonl(output_path)
    by_path = {str(item.get("path")): item for item in existing if item.get("path")}

    merged = 0
    skipped = 0
    for item in incoming:
        if item["path"] in by_path and not overwrite:
            skipped += 1
            continue
        by_path[item["path"]] = item
        merged += 1

    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path(".") else None
    with output_path.open("w", encoding="utf-8") as f:
        for item in by_path.values():
            item.pop("_line_number", None)
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
            f.write("\n")
    return {"output": str(output_path), "merged": merged, "skipped_existing": skipped, "total_records": len(by_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and ingest external multimodal captioning tasks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_cmd = subparsers.add_parser("export", help="Export uncaptioned images as external annotation batches.")
    export_cmd.add_argument("--images-dir", default=str(IMAGES_DIR), help="Local image corpus directory.")
    export_cmd.add_argument("--output-dir", default="", help="Batch output directory. Defaults to annotation_batches/<timestamp>.")
    export_cmd.add_argument("--caption-index", default=str(CAPTION_INDEX_PATH), help="Existing caption JSONL to skip.")
    export_cmd.add_argument("--batch-size", type=int, default=50, help="Images per batch JSONL.")
    export_cmd.add_argument("--limit", type=int, default=0, help="Export at most N images. 0 means no limit.")
    export_cmd.add_argument("--include-captioned", action="store_true", help="Do not skip paths already in caption index.")

    validate_cmd = subparsers.add_parser("validate", help="Validate an external captions JSONL file.")
    validate_cmd.add_argument("input", help="External captions JSONL.")
    validate_cmd.add_argument("--source-model", default="external-multimodal", help="Model/source label for validation.")

    merge_cmd = subparsers.add_parser("merge", help="Merge external captions into image_index.jsonl.")
    merge_cmd.add_argument("input", help="External captions JSONL.")
    merge_cmd.add_argument("--output", default=str(CAPTION_INDEX_PATH), help="Caption index JSONL to update.")
    merge_cmd.add_argument("--source-model", default="external-multimodal", help="Model/source label for merged records.")
    merge_cmd.add_argument("--overwrite", action="store_true", help="Replace existing captions for the same path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "export":
        output_dir = Path(args.output_dir).expanduser() if args.output_dir else ANNOTATION_BATCHES_DIR / time.strftime("%Y%m%d-%H%M%S")
        summary = export_batches(
            images_dir=Path(args.images_dir).expanduser(),
            output_dir=output_dir,
            caption_index=Path(args.caption_index).expanduser(),
            batch_size=max(1, args.batch_size),
            limit=max(0, args.limit),
            include_captioned=args.include_captioned,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "validate":
        summary = validate_external_captions(Path(args.input).expanduser(), source_model=args.source_model)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if not summary["errors"] else 1

    if args.command == "merge":
        validation = validate_external_captions(Path(args.input).expanduser(), source_model=args.source_model)
        if validation["errors"]:
            print(json.dumps(validation, ensure_ascii=False, indent=2))
            return 1
        summary = merge_external_captions(
            input_path=Path(args.input).expanduser(),
            output_path=Path(args.output).expanduser(),
            source_model=args.source_model,
            overwrite=args.overwrite,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
