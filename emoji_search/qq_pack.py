"""Inspect and import QQ Chat Exporter sticker packs into the local corpus."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from .config import IMAGE_EXTENSIONS, IMAGES_DIR, MANIFEST_PATH


def load_pack_info(pack_dir: Path) -> dict[str, Any]:
    pack_info_path = pack_dir / "pack_info.json"
    if not pack_info_path.exists():
        raise FileNotFoundError(f"Missing pack_info.json: {pack_info_path}")
    with pack_info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_pack_files(pack_dir: Path) -> list[Path]:
    stickers_dir = pack_dir / "stickers"
    if not stickers_dir.exists():
        return []
    return sorted(
        path
        for path in stickers_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_sticker_file_lookup(pack_dir: Path, source_dirs: list[Path] | None = None) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    search_dirs = [pack_dir / "stickers", *(source_dirs or [])]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for path in sorted(
            path
            for path in search_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            lookup.setdefault(path.name.lower(), path)
            lookup.setdefault(path.stem.upper(), path)
    return lookup


def source_path_for_sticker(sticker: dict[str, Any], pack_dir: Path, lookup: dict[str, Path]) -> Path | None:
    raw_path = str(sticker.get("path") or "")
    if raw_path:
        candidate = Path(raw_path).expanduser()
        if candidate.is_file():
            return candidate
        pack_candidate = pack_dir / "stickers" / candidate.name
        if pack_candidate.is_file():
            return pack_candidate
        by_name = lookup.get(candidate.name.lower())
        if by_name:
            return by_name

    md5 = str(sticker.get("md5") or "").upper()
    if md5:
        return lookup.get(md5)
    return None


def safe_fragment(value: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._")
    return value or fallback


def destination_name(sticker: dict[str, Any], source_path: Path) -> str:
    sticker_id = safe_fragment(str(sticker.get("stickerId") or ""), "unknown")
    if sticker_id.isdigit():
        sticker_id = f"{int(sticker_id):04d}"
    md5 = safe_fragment(str(sticker.get("md5") or source_path.stem), source_path.stem)
    suffix = source_path.suffix.lower() or ".jpg"
    return f"qq_favorite_{sticker_id}_{md5[:16]}{suffix}"


def relative_image_path(path: Path, images_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(images_dir.resolve().parent))
    except ValueError:
        return str(Path(images_dir.name) / path.name)


def inspect_pack(pack_dir: Path, source_dirs: list[Path] | None = None) -> dict[str, Any]:
    pack = load_pack_info(pack_dir)
    stickers = list(pack.get("stickers") or [])
    lookup = build_sticker_file_lookup(pack_dir, source_dirs)
    sources = [source_path_for_sticker(sticker, pack_dir, lookup) for sticker in stickers]
    available = [path for path in sources if path is not None]
    missing_examples = [
        {
            "stickerId": sticker.get("stickerId"),
            "name": sticker.get("name"),
            "path": sticker.get("path"),
            "md5": sticker.get("md5"),
        }
        for sticker, source in zip(stickers, sources, strict=False)
        if source is None
    ][:10]
    return {
        "pack_dir": str(pack_dir),
        "source_dirs": [str(path) for path in source_dirs or []],
        "pack_name": pack.get("packName"),
        "declared_sticker_count": pack.get("stickerCount"),
        "metadata_records": len(stickers),
        "files_under_stickers_dir": len(iter_pack_files(pack_dir)),
        "available_source_files": len(available),
        "missing_source_files": len(stickers) - len(available),
        "downloaded_records": sum(1 for sticker in stickers if sticker.get("downloaded")),
        "positive_file_size_records": sum(1 for sticker in stickers if int(sticker.get("fileSize") or 0) > 0),
        "missing_examples": missing_examples,
    }


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


def import_pack(
    *,
    pack_dir: Path,
    source_dirs: list[Path],
    images_dir: Path,
    metadata_output: Path,
    mode: str,
    overwrite: bool,
    dry_run: bool,
    limit: int,
) -> dict[str, Any]:
    pack = load_pack_info(pack_dir)
    stickers = list(pack.get("stickers") or [])
    lookup = build_sticker_file_lookup(pack_dir, source_dirs)
    manifest_items: list[dict[str, Any]] = []
    imported = 0
    skipped_existing = 0
    missing = 0

    for sticker in stickers:
        if limit and imported >= limit:
            break
        source = source_path_for_sticker(sticker, pack_dir, lookup)
        if source is None:
            missing += 1
            continue

        destination = images_dir / destination_name(sticker, source)
        action = "dry-run"
        if not dry_run:
            action = copy_or_link(source, destination, mode, overwrite)
        if action == "exists":
            skipped_existing += 1
        else:
            imported += 1

        manifest_items.append(
            {
                "path": relative_image_path(destination, images_dir),
                "source_path": str(source),
                "pack_dir": str(pack_dir),
                "pack_name": pack.get("packName"),
                "sticker_id": sticker.get("stickerId"),
                "name": sticker.get("name"),
                "md5": sticker.get("md5"),
                "mode": mode,
                "action": action,
            }
        )

    if manifest_items and not dry_run:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        with metadata_output.open("a", encoding="utf-8") as f:
            for item in manifest_items:
                f.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
                f.write("\n")

    return {
        "pack_dir": str(pack_dir),
        "source_dirs": [str(path) for path in source_dirs],
        "images_dir": str(images_dir),
        "metadata_output": str(metadata_output),
        "mode": mode,
        "dry_run": dry_run,
        "imported_or_planned": imported,
        "skipped_existing": skipped_existing,
        "missing_source_files": missing,
        "manifest_records": len(manifest_items),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect or import QQ Chat Exporter sticker packs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = subparsers.add_parser("inspect", help="Report whether a sticker pack has usable image files.")
    inspect_cmd.add_argument("--pack-dir", required=True, help="QQ exporter sticker pack directory.")
    inspect_cmd.add_argument(
        "--source-dir",
        action="append",
        default=[],
        help="Additional directory to search for source files by md5/name. Can be repeated.",
    )
    inspect_cmd.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    import_cmd = subparsers.add_parser("import", help="Copy/link available sticker files into the local images corpus.")
    import_cmd.add_argument("--pack-dir", required=True, help="QQ exporter sticker pack directory.")
    import_cmd.add_argument(
        "--source-dir",
        action="append",
        default=[],
        help="Additional directory to search for source files by md5/name. Can be repeated.",
    )
    import_cmd.add_argument("--images-dir", default=str(IMAGES_DIR), help="Destination corpus directory.")
    import_cmd.add_argument("--metadata-output", default=str(MANIFEST_PATH), help="Append-only JSONL import manifest.")
    import_cmd.add_argument("--mode", choices=("copy", "symlink", "hardlink"), default="copy", help="Import strategy.")
    import_cmd.add_argument("--overwrite", action="store_true", help="Replace existing destination files.")
    import_cmd.add_argument("--dry-run", action="store_true", help="Show what would be imported without writing files.")
    import_cmd.add_argument("--limit", type=int, default=0, help="Import at most N available files. 0 means no limit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "inspect":
        summary = inspect_pack(
            Path(args.pack_dir).expanduser(),
            [Path(path).expanduser() for path in args.source_dir],
        )
        if args.json:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            for key, value in summary.items():
                print(f"{key}: {value}")
        return 0

    if args.command == "import":
        summary = import_pack(
            pack_dir=Path(args.pack_dir).expanduser(),
            source_dirs=[Path(path).expanduser() for path in args.source_dir],
            images_dir=Path(args.images_dir).expanduser(),
            metadata_output=Path(args.metadata_output).expanduser(),
            mode=args.mode,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            limit=args.limit,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["imported_or_planned"] or summary["missing_source_files"] == 0 else 1

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
