"""Caption visual-cluster representatives with the configured vision API."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import CAPTION_INDEX_PATH, CLUSTER_CSV_PATH, IMAGES_DIR, PROJ_ROOT
from .envfile import api_settings_from_env, load_env_file


ENV_PATH = PROJ_ROOT / ".env"


def read_cluster_rows(cluster_path: Path) -> list[dict[str, str]]:
    if not cluster_path.exists():
        return []
    with cluster_path.open("r", encoding="utf-8", newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("path") and row.get("visual_cluster_id")]


def load_done_paths(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    done: set[str] = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("ok") and item.get("path"):
                done.add(str(item["path"]))
    return done


def cluster_size(row: dict[str, str]) -> int:
    try:
        return int(row.get("cluster_size") or "1")
    except ValueError:
        return 1


def select_representatives(
    rows: list[dict[str, str]],
    *,
    done_paths: set[str],
    overwrite: bool,
    min_cluster_size: int,
    asset_types: set[str],
    limit: int,
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if cluster_size(row) < min_cluster_size:
            continue
        if asset_types and row.get("asset_type") not in asset_types:
            continue
        grouped[row["visual_cluster_id"]].append(row)

    representatives: list[dict[str, str]] = []
    for _cluster_id, cluster_rows in sorted(
        grouped.items(),
        key=lambda item: (-max(cluster_size(row) for row in item[1]), item[0]),
    ):
        cluster_rows = sorted(cluster_rows, key=lambda row: row.get("path", ""))
        representative = cluster_rows[0]
        if not overwrite and representative["path"] in done_paths:
            continue
        representatives.append(representative)
        if limit and len(representatives) >= limit:
            break
    return representatives


def image_path_for_row(row: dict[str, str], images_dir: Path) -> Path:
    rel_path = Path(row["path"])
    if rel_path.is_absolute():
        return rel_path
    images_root = images_dir.resolve()
    project_root = images_root.parent
    return project_root / rel_path


def append_jsonl(output_path: Path, item: dict[str, Any]) -> None:
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def caption_representatives(args: argparse.Namespace) -> int:
    from main import describe_image, summarize_failure

    load_env_file(ENV_PATH)
    settings = api_settings_from_env(ENV_PATH)
    if not settings["api_key"]:
        print(f"Missing API key in {ENV_PATH}.")
        return 2

    images_dir = Path(args.images_dir).expanduser()
    cluster_path = Path(args.clusters).expanduser()
    output_path = Path(args.output).expanduser()
    rows = read_cluster_rows(cluster_path)
    done_paths = set() if args.overwrite else load_done_paths(output_path)
    asset_types = {item.strip() for item in args.asset_type for item in item.split(",") if item.strip()}
    representatives = select_representatives(
        rows,
        done_paths=done_paths,
        overwrite=args.overwrite,
        min_cluster_size=max(1, args.min_cluster_size),
        asset_types=asset_types,
        limit=max(0, args.limit),
    )

    endpoint = settings["base_url"].rstrip("/") + "/chat/completions"
    print(f"Cluster rows: {len(rows)}")
    print(f"Representatives pending: {len(representatives)}")
    print(f"Output: {output_path}")
    print(f"Endpoint: {endpoint}")
    print(f"Model: {settings['model']}")

    if args.dry_run:
        for row in representatives:
            print(
                f"{row['path']}\t{row.get('visual_cluster_id', '')}\t"
                f"size={row.get('cluster_size', '')}\tasset={row.get('asset_type', '')}"
            )
        return 0

    if not representatives:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path(".") else None
    started = time.time()
    completed = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_row = {
            executor.submit(
                describe_image,
                image_path_for_row(row, images_dir),
                row["path"],
                settings["base_url"].rstrip("/"),
                settings["api_key"],
                settings["model"],
                args.detail,
                args.max_tokens,
                not args.no_response_format,
                args.timeout,
                args.retries,
            ): row
            for row in representatives
        }
        for future in concurrent.futures.as_completed(future_to_row):
            row = future_to_row[future]
            try:
                item = future.result()
            except Exception as exc:
                item = {
                    "path": row["path"],
                    "ok": False,
                    "model": settings["model"],
                    "error": str(exc),
                }
            item["cluster"] = {
                "visual_cluster_id": row.get("visual_cluster_id", ""),
                "cluster_size": row.get("cluster_size", ""),
                "asset_type": row.get("asset_type", ""),
            }
            item["source"] = "qwen_cluster_representative"
            append_jsonl(output_path, item)
            completed += 1
            failed += 0 if item.get("ok") else 1
            status = "ok" if item.get("ok") else "failed"
            print(f"[{completed}/{len(representatives)}] {status}: {item['path']}", flush=True)
            if args.verbose_failures and not item.get("ok"):
                print("  " + summarize_failure(item), flush=True)

    print(f"Done. Success: {completed - failed}. Failed: {failed}. Elapsed: {time.time() - started:.1f}s")
    return 0 if failed == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caption one representative image per visual cluster.")
    parser.add_argument("--images-dir", default=str(IMAGES_DIR), help="Local image corpus directory.")
    parser.add_argument("--clusters", default=str(CLUSTER_CSV_PATH), help="Cluster CSV from emoji-cluster.")
    parser.add_argument("--output", default=str(CAPTION_INDEX_PATH), help="Caption JSONL output.")
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent API requests.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N pending representatives.")
    parser.add_argument("--min-cluster-size", type=int, default=1, help="Only process clusters of at least this size.")
    parser.add_argument(
        "--asset-type",
        action="append",
        default=[],
        help="Optional asset_type filter. Can be repeated or comma-separated.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Reprocess representatives already captioned.")
    parser.add_argument("--detail", default="low", choices=("low", "high", "auto"), help="Vision detail hint.")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum response tokens per image.")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=1, help="Retries per image after failures.")
    parser.add_argument(
        "--no-response-format",
        action="store_true",
        help="Do not request response_format=json_object.",
    )
    parser.add_argument("--verbose-failures", action="store_true", help="Print server details for failed images.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected representatives without API calls.")
    return parser.parse_args()


def main() -> int:
    return caption_representatives(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
