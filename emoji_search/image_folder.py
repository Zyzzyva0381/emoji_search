"""Import any folder of image files into the local emoji corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .config import IMAGE_EXTENSIONS, IMAGES_DIR, MANIFEST_PATH


def iter_images(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_fragment(value: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._")
    return value or fallback


def relative_image_path(path: Path, images_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(images_dir.resolve().parent))
    except ValueError:
        return str(Path(images_dir.name) / path.name)


def read_manifest(path: Path) -> list[dict[str, Any]]:
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


def destination_name(source_path: Path, sha256: str, prefix: str) -> str:
    stem = safe_fragment(source_path.stem, "image")
    suffix = source_path.suffix.lower() or ".png"
    return f"{safe_fragment(prefix, 'emoji')}_{stem}_{sha256[:16]}{suffix}"


def copy_or_link(source: Path, destination: Path, mode: str, overwrite: bool) -> str:
    if destination.exists():
        if not overwrite:
            return "exists"
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        os.symlink(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        raise ValueError(f"Unsupported import mode: {mode}")
    return "imported"


def inspect_folder(source_dir: Path) -> dict[str, Any]:
    images = iter_images(source_dir)
    extensions = Counter(path.suffix.lower() for path in images)
    total_bytes = sum(path.stat().st_size for path in images)
    hashes = Counter(file_sha256(path) for path in images)
    return {
        "source_dir": str(source_dir),
        "image_files": len(images),
        "total_bytes": total_bytes,
        "total_megabytes": round(total_bytes / 1024 / 1024, 3),
        "extensions": dict(sorted(extensions.items())),
        "duplicate_content_groups": sum(1 for count in hashes.values() if count > 1),
        "duplicate_files": sum(count - 1 for count in hashes.values() if count > 1),
    }


def import_folder(
    *,
    source_dir: Path,
    images_dir: Path,
    metadata_output: Path,
    mode: str,
    overwrite: bool,
    dry_run: bool,
    keep_duplicates: bool,
    prefix: str,
    collection: str,
    limit: int,
) -> dict[str, Any]:
    existing_manifest = read_manifest(metadata_output)
    existing_by_hash = {
        str(item.get("sha256")): item
        for item in existing_manifest
        if item.get("sha256") and item.get("path")
    }
    seen_hashes = set(existing_by_hash)
    manifest_items: list[dict[str, Any]] = []
    imported = 0
    skipped_duplicate = 0
    skipped_existing = 0

    for source in iter_images(source_dir):
        if limit and imported >= limit:
            break
        sha256 = file_sha256(source)
        if not keep_duplicates and sha256 in seen_hashes:
            skipped_duplicate += 1
            continue

        destination = images_dir / destination_name(source, sha256, prefix)
        action = "dry-run"
        if not dry_run:
            action = copy_or_link(source, destination, mode, overwrite)
        if action == "exists":
            skipped_existing += 1
        else:
            imported += 1
        seen_hashes.add(sha256)

        stat = source.stat()
        manifest_items.append(
            {
                "path": relative_image_path(destination, images_dir),
                "source_path": str(source),
                "source_root": str(source_dir),
                "relative_source_path": str(source.relative_to(source_dir)),
                "original_name": source.name,
                "collection": collection,
                "sha256": sha256,
                "file_size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "suffix": source.suffix.lower(),
                "mode": mode,
                "action": action,
                "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        )

    if manifest_items and not dry_run:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        with metadata_output.open("a", encoding="utf-8") as f:
            for item in manifest_items:
                f.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
                f.write("\n")

    return {
        "source_dir": str(source_dir),
        "images_dir": str(images_dir),
        "metadata_output": str(metadata_output),
        "mode": mode,
        "dry_run": dry_run,
        "imported_or_planned": imported,
        "skipped_duplicate": skipped_duplicate,
        "skipped_existing": skipped_existing,
        "manifest_records": len(manifest_items),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or import any image folder into the emoji corpus.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = subparsers.add_parser("inspect", help="Report image counts, extensions, size, and duplicate files.")
    inspect_cmd.add_argument("source_dir", help="Source image folder.")

    import_cmd = subparsers.add_parser("import", help="Copy/link images into the local corpus and write a manifest.")
    import_cmd.add_argument("source_dir", help="Source image folder.")
    import_cmd.add_argument("--images-dir", default=str(IMAGES_DIR), help="Destination corpus directory.")
    import_cmd.add_argument("--metadata-output", default=str(MANIFEST_PATH), help="Append-only JSONL import manifest.")
    import_cmd.add_argument("--mode", choices=("copy", "symlink", "hardlink"), default="copy", help="Import strategy.")
    import_cmd.add_argument("--overwrite", action="store_true", help="Replace existing destination files.")
    import_cmd.add_argument("--dry-run", action="store_true", help="Show what would be imported without writing files.")
    import_cmd.add_argument("--keep-duplicates", action="store_true", help="Import files even if the same sha256 exists.")
    import_cmd.add_argument("--prefix", default="emoji", help="Destination filename prefix.")
    import_cmd.add_argument("--collection", default="", help="Human-readable collection/source name.")
    import_cmd.add_argument("--limit", type=int, default=0, help="Import at most N files. 0 means no limit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "inspect":
        summary = inspect_folder(Path(args.source_dir).expanduser())
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "import":
        source_dir = Path(args.source_dir).expanduser()
        summary = import_folder(
            source_dir=source_dir,
            images_dir=Path(args.images_dir).expanduser(),
            metadata_output=Path(args.metadata_output).expanduser(),
            mode=args.mode,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            keep_duplicates=args.keep_duplicates,
            prefix=args.prefix,
            collection=args.collection or source_dir.name,
            limit=max(0, args.limit),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
