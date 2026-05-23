"""Interactive command line workflow for local emoji search setup."""

from __future__ import annotations

import concurrent.futures
import getpass
import json
import sys
import time
from pathlib import Path
from typing import Any

from .caption_schema import FIELD_NAMES
from .config import CAPTION_INDEX_PATH, IMAGE_EXTENSIONS, IMAGES_DIR, MANIFEST_PATH, PROJ_ROOT, VECTOR_INDEX_PATH, VECTOR_MODEL
from .envfile import api_settings_from_env, load_env_file, update_env_file
from .image_folder import import_folder, inspect_folder


ENV_PATH = PROJ_ROOT / ".env"


def prompt_text(label: str, default: str = "", *, secret: bool = False) -> str:
    suffix = f" [{default}]" if default and not secret else ""
    prompt = f"{label}{suffix}: "
    value = getpass.getpass(prompt) if secret else input(prompt)
    value = value.strip()
    return value or default


def prompt_bool(label: str, default: bool = True) -> bool:
    marker = "Y/n" if default else "y/N"
    value = input(f"{label} [{marker}]: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes", "1", "true", "是"}


def prompt_int(label: str, default: int, *, minimum: int = 0) -> int:
    while True:
        value = input(f"{label} [{default}]: ").strip()
        if not value:
            return default
        try:
            parsed = int(value)
        except ValueError:
            print("请输入整数。")
            continue
        if parsed < minimum:
            print(f"不能小于 {minimum}。")
            continue
        return parsed


def print_json(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def configure_api() -> dict[str, str]:
    settings = api_settings_from_env(ENV_PATH)
    print("\nAPI 配置写入 .env。请求仍走 OpenAI-compatible /chat/completions。")
    base_url = prompt_text("API base URL", settings["base_url"])
    model = prompt_text("Vision model", settings["model"])
    current_key = settings["api_key"]
    key_label = "API token"
    if current_key:
        key_label += "（已存在，回车保留）"
    api_key = prompt_text(key_label, "", secret=True) or current_key
    if not api_key:
        print("未写入 token；caption 步骤会失败。")

    updates = {
        "EMOJI_API_BASE_URL": base_url.rstrip("/"),
        "EMOJI_API_KEY": api_key,
        "EMOJI_API_MODEL": model,
        "OPENAI_BASE_URL": base_url.rstrip("/"),
        "OPENAI_API_KEY": api_key,
        "OPENAI_MODEL": model,
    }
    update_env_file(ENV_PATH, updates)
    load_env_file(ENV_PATH, override=True)
    print(f"已更新 {ENV_PATH}。")
    return api_settings_from_env(ENV_PATH)


def import_images_interactive() -> None:
    default_source = str(IMAGES_DIR)
    source = Path(prompt_text("\n图片文件夹路径", default_source)).expanduser()
    if not source.exists():
        print(f"目录不存在：{source}")
        return
    summary = inspect_folder(source)
    print_json(summary)
    if summary["image_files"] == 0:
        return
    if not prompt_bool("导入这些图片到项目 images/ 目录？", True):
        return

    mode = prompt_text("导入方式 copy/hardlink/symlink", "copy")
    if mode not in {"copy", "hardlink", "symlink"}:
        print("导入方式无效，使用 copy。")
        mode = "copy"
    prefix = prompt_text("目标文件名前缀", "emoji")
    collection = prompt_text("集合名称", source.name)
    result = import_folder(
        source_dir=source,
        images_dir=IMAGES_DIR,
        metadata_output=MANIFEST_PATH,
        mode=mode,
        overwrite=False,
        dry_run=False,
        keep_duplicates=False,
        prefix=prefix,
        collection=collection,
        limit=0,
    )
    print_json(result)


def pending_caption_tasks(images_dir: Path, output_path: Path, overwrite: bool, limit: int) -> list[Path]:
    from main import iter_images, load_done_paths

    images = iter_images(images_dir)
    done_paths = set() if overwrite else load_done_paths(output_path)
    tasks = [
        path
        for path in images
        if str(path.relative_to(images_dir.parent)) not in done_paths
    ]
    return tasks[:limit] if limit else tasks


def caption_images_polling() -> int:
    from main import append_jsonl, describe_image, summarize_failure

    settings = api_settings_from_env(ENV_PATH)
    if not settings["api_key"]:
        print("\n.env 中没有 API token。先进入菜单 1 配置 API。")
        return 2

    images_dir = IMAGES_DIR
    output_path = CAPTION_INDEX_PATH
    if not images_dir.exists():
        print(f"图片目录不存在：{images_dir}")
        return 2

    workers = prompt_int("\n并发请求数", 2, minimum=1)
    limit = prompt_int("本轮最多处理多少张，0 表示全部", 0, minimum=0)
    overwrite = prompt_bool("重新处理已成功 caption 的图片？", False)
    detail = prompt_text("Vision detail low/high/auto", "low")
    if detail not in {"low", "high", "auto"}:
        detail = "low"
    max_tokens = prompt_int("每张最大输出 tokens", 500, minimum=64)
    timeout = prompt_int("单次请求超时秒数", 120, minimum=10)
    retries = prompt_int("失败重试次数", 1, minimum=0)
    use_response_format = prompt_bool("请求 response_format=json_object？", True)
    verbose_failures = prompt_bool("失败时打印服务端详情？", False)

    tasks = pending_caption_tasks(images_dir, output_path, overwrite, limit)
    print(f"\n图片目录：{images_dir}")
    print(f"输出索引：{output_path}")
    print(f"API endpoint：{settings['base_url'].rstrip('/')}/chat/completions")
    print(f"模型：{settings['model']}")
    print(f"待处理：{len(tasks)} 张")
    if not tasks:
        return 0
    if not prompt_bool("开始调用 API caption？", True):
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed = 0
    failed = 0
    last_report = 0.0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(
                describe_image,
                path,
                str(path.relative_to(images_dir.parent)),
                settings["base_url"].rstrip("/"),
                settings["api_key"],
                settings["model"],
                detail,
                max_tokens,
                use_response_format,
                timeout,
                retries,
            ): path
            for path in tasks
        }
        while pending:
            done, _ = concurrent.futures.wait(
                pending,
                timeout=0.5,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            now = time.time()
            if not done and now - last_report >= 5:
                print(f"[poll] completed={completed} failed={failed} pending={len(pending)} elapsed={now - started:.1f}s")
                last_report = now
                continue
            for future in done:
                path = pending.pop(future)
                try:
                    item = future.result()
                except Exception as exc:
                    item = {
                        "path": str(path.relative_to(images_dir.parent)),
                        "ok": False,
                        "model": settings["model"],
                        "error": str(exc),
                    }
                append_jsonl(output_path, item)
                completed += 1
                failed += 0 if item.get("ok") else 1
                status = "ok" if item.get("ok") else "failed"
                print(f"[{completed}/{len(tasks)}] {status}: {item['path']}", flush=True)
                if verbose_failures and not item.get("ok"):
                    print("  " + summarize_failure(item), flush=True)

    print(f"Done. Success: {completed - failed}. Failed: {failed}. Elapsed: {time.time() - started:.1f}s")
    return 0 if failed == 0 else 1


def build_vector_index_interactive() -> int:
    if not CAPTION_INDEX_PATH.exists():
        print(f"\ncaption 索引不存在：{CAPTION_INDEX_PATH}")
        return 2
    model = prompt_text("\nEmbedding model", VECTOR_MODEL)
    batch_size = prompt_int("Embedding batch size", 32, minimum=1)
    if not prompt_bool("开始构建向量索引？", True):
        return 0
    from semantic_search import build_index

    build_index(
        input_path=CAPTION_INDEX_PATH,
        output_path=VECTOR_INDEX_PATH,
        model_name=model,
        fields=FIELD_NAMES,
        batch_size=batch_size,
    )
    return 0


def show_status() -> None:
    from main import load_done_paths

    image_count = (
        len([path for path in IMAGES_DIR.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS])
        if IMAGES_DIR.exists()
        else 0
    )
    caption_count = len(load_done_paths(CAPTION_INDEX_PATH)) if CAPTION_INDEX_PATH.exists() else 0
    settings = api_settings_from_env(ENV_PATH)
    print("\n当前状态")
    print(f"- images dir: {IMAGES_DIR}")
    print(f"- image files: {image_count}")
    print(f"- caption index: {CAPTION_INDEX_PATH} ({caption_count} ok)")
    print(f"- vector index: {VECTOR_INDEX_PATH} ({'exists' if VECTOR_INDEX_PATH.exists() else 'missing'})")
    print(f"- API base URL: {settings['base_url']}")
    print(f"- API model: {settings['model']}")
    print(f"- API token: {'configured' if settings['api_key'] else 'missing'}")


def main() -> int:
    load_env_file(ENV_PATH)
    while True:
        show_status()
        print(
            "\n选择操作：\n"
            "1. 配置 API 到 .env\n"
            "2. 导入任意图片文件夹\n"
            "3. 调用 API caption 图片（轮询进度）\n"
            "4. 构建向量索引\n"
            "5. 完整流程：导入 -> caption -> 建索引\n"
            "q. 退出"
        )
        choice = input("> ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return 0
        if choice == "1":
            configure_api()
        elif choice == "2":
            import_images_interactive()
        elif choice == "3":
            caption_images_polling()
        elif choice == "4":
            build_vector_index_interactive()
        elif choice == "5":
            import_images_interactive()
            if caption_images_polling() == 0:
                build_vector_index_interactive()
        else:
            print("未知选项。")


if __name__ == "__main__":
    raise SystemExit(main())
