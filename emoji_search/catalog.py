"""Export and apply human-editable caption/tag catalogs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .caption_schema import FIELD_NAMES, caption_index_item, normalize_caption, normalize_text, semantic_text
from .config import CAPTION_INDEX_PATH, CLUSTER_CSV_PATH, IMAGE_EXTENSIONS, IMAGES_DIR, MANIFEST_PATH, PROJ_ROOT


CATALOG_COLUMNS = [
    "path",
    "file_name",
    "source_path",
    "asset_type",
    "asset_confidence",
    "visual_cluster_id",
    "near_duplicate_group",
    "cluster_size",
    "duplicate_group_size",
    *FIELD_NAMES,
]


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def load_by_path(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("path")): item
        for item in read_jsonl(path)
        if item.get("path")
    }


def load_csv_by_path(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {
            str(row.get("path")): row
            for row in csv.DictReader(f)
            if row.get("path")
        }


def export_catalog(
    *,
    images_dir: Path,
    caption_index: Path,
    manifest_path: Path,
    cluster_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    captions = load_by_path(caption_index)
    manifest = load_by_path(manifest_path)
    clusters = load_csv_by_path(cluster_path)
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path(".") else None
    rows = 0
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CATALOG_COLUMNS)
        writer.writeheader()
        for image_path in iter_images(images_dir):
            rel = relative_image_path(image_path, images_dir)
            caption = normalize_caption((captions.get(rel, {}).get("caption") or {}), fill_missing="")
            row = {
                "path": rel,
                "file_name": image_path.name,
                "source_path": manifest.get(rel, {}).get("source_path", ""),
                "asset_type": clusters.get(rel, {}).get("asset_type", ""),
                "asset_confidence": clusters.get(rel, {}).get("asset_confidence", ""),
                "visual_cluster_id": clusters.get(rel, {}).get("visual_cluster_id", ""),
                "near_duplicate_group": clusters.get(rel, {}).get("near_duplicate_group", ""),
                "cluster_size": clusters.get(rel, {}).get("cluster_size", ""),
                "duplicate_group_size": clusters.get(rel, {}).get("duplicate_group_size", ""),
                **caption,
            }
            writer.writerow(row)
            rows += 1
    return {"output": str(output_path), "rows": rows}


def apply_catalog(
    *,
    input_path: Path,
    caption_index: Path,
    overwrite: bool,
) -> dict[str, Any]:
    existing = load_by_path(caption_index)
    changed = 0
    created = 0
    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = normalize_text(row.get("path"))
            if not path:
                continue
            incoming = {
                field: normalize_text(row.get(field))
                for field in FIELD_NAMES
            }
            if path not in existing:
                existing[path] = caption_index_item(
                    path=path,
                    caption=incoming,
                    model="manual-catalog",
                    endpoint="manual",
                    source="catalog_csv",
                )
                created += 1
                continue

            caption = normalize_caption(existing[path].get("caption") or {}, fill_missing="")
            for field, value in incoming.items():
                if overwrite or value:
                    caption[field] = value
            existing[path]["caption"] = normalize_caption(caption)
            existing[path]["semantic_text"] = semantic_text(existing[path]["caption"])
            existing[path]["source"] = str(existing[path].get("source") or "catalog_csv")
            changed += 1

    caption_index.parent.mkdir(parents=True, exist_ok=True) if caption_index.parent != Path(".") else None
    with caption_index.open("w", encoding="utf-8") as f:
        for item in existing.values():
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
            f.write("\n")
    return {"output": str(caption_index), "created": created, "changed": changed, "total_records": len(existing)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export/apply a human-editable CSV catalog for captions and tags.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_cmd = subparsers.add_parser("export", help="Export a CSV catalog for human review and manual tags.")
    export_cmd.add_argument("--images-dir", default=str(IMAGES_DIR), help="Local image corpus directory.")
    export_cmd.add_argument("--caption-index", default=str(CAPTION_INDEX_PATH), help="Caption index JSONL.")
    export_cmd.add_argument("--manifest", default=str(MANIFEST_PATH), help="Import manifest JSONL.")
    export_cmd.add_argument("--clusters", default=str(CLUSTER_CSV_PATH), help="Cluster CSV produced by emoji-cluster.")
    export_cmd.add_argument("--output", default=str(PROJ_ROOT / "emoji_catalog.csv"), help="CSV catalog path.")

    apply_cmd = subparsers.add_parser("apply", help="Apply edited CSV fields back into image_index.jsonl.")
    apply_cmd.add_argument("input", help="Edited CSV catalog path.")
    apply_cmd.add_argument("--caption-index", default=str(CAPTION_INDEX_PATH), help="Caption index JSONL to update.")
    apply_cmd.add_argument("--overwrite", action="store_true", help="Overwrite existing fields with blank CSV cells too.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "export":
        summary = export_catalog(
            images_dir=Path(args.images_dir).expanduser(),
            caption_index=Path(args.caption_index).expanduser(),
            manifest_path=Path(args.manifest).expanduser(),
            cluster_path=Path(args.clusters).expanduser(),
            output_path=Path(args.output).expanduser(),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "apply":
        summary = apply_catalog(
            input_path=Path(args.input).expanduser(),
            caption_index=Path(args.caption_index).expanduser(),
            overwrite=args.overwrite,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
