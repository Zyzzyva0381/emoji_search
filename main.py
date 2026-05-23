#!/usr/bin/env python3
"""Build a semantic caption index for emoji images via a chat-completions API.

This script does not inspect images locally. It only reads image files, sends
them to a compatible /v1/chat/completions endpoint, and stores the model's
brief structured description for later semantic search/indexing.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import random
import sys
import time
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from emoji_search.caption_schema import FIELD_NAMES, parse_caption_json
from emoji_search.config import PROJ_ROOT
from emoji_search.envfile import api_settings_from_env


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
}

DEFAULT_PROMPT = """请为这个表情包图片生成简短、利于语义检索的**中文**结构化描述。

需要覆盖这些字段：
- image_composition: 表情包中图片构成
- character_name: 角色的名称
- expression: 表情
- action: 动作
- subjective_emotion: 主观情绪
- text_in_image: 图中文字
- usage_context: 适合在聊天中的什么场景使用
- wechat_keyword: 微信表情含义词候选，不超过 4 个汉字
- manual_tags: 人工标签，自动标注时输出 NONE
- notes: 其他有助于检索的极短补充

若某一项缺失或无法判断，输出占位符“NONE”。

一些模型可能无法认出的角色特征：
1. 绿色头发、蓝色眼睛、黄色三角星是角色 Suzume 的特征，该角色一般穿着有猫耳的卫衣。
2. 蓝色双马尾是初音未来的特征。
3. 灰色头发和 8 字形发辫是洛天依的特征。
4. 猫猫虫 Capoo 是又像虫又像猫的生物，有六条腿，图片可能不会把腿显示全，背上可能有三条条纹，肚子是白色的。
5. 萨卡班甲鱼是一种类似鱼的生物，背部是灰色，腹部是白色，一般有圆形的眼睛和倒三角形的嘴巴。
6. 白色/浅色头发、有猫耳、圆脸、可能有双马尾的猫娘角色是明风。

如果表情包中有无法识别的角色、物品，或者以上提到的某项特征难以用语言描述，直接忽略或用占位符代替，不要写冗余的无关文字。

只输出一个 JSON 对象，不要 Markdown，不要解释。字段值都使用简短中文，必须是字符串，不要输出数组。"""


class ApiRequestError(Exception):
    """Error raised for a failed API request with server response details."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        reason: str | None = None,
        headers: dict[str, str] | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.headers = headers or {}
        self.body = body

    def to_log(self) -> dict[str, Any]:
        return {
            "message": str(self),
            "status": self.status,
            "reason": self.reason,
            "headers": self.headers,
            "body": self.body,
        }


def parse_args() -> argparse.Namespace:
    api_settings = api_settings_from_env(PROJ_ROOT / ".env")
    parser = argparse.ArgumentParser(
        description="Index images semantically by captioning them with a chat-completions vision API.",
    )
    parser.add_argument("--images-dir", default="images", help="Directory containing images.")
    parser.add_argument("--output", default="image_index.jsonl", help="Output JSONL file.")
    parser.add_argument(
        "--base-url",
        default=api_settings["base_url"],
        help="API base URL, for example https://api.openai.com/v1.",
    )
    parser.add_argument(
        "--api-key",
        default=api_settings["api_key"],
        help="API key. Prefer setting OPENAI_API_KEY or API_KEY in the environment.",
    )
    parser.add_argument(
        "--model",
        default=api_settings["model"],
        help="Vision-capable chat model name used by your API provider.",
    )
    parser.add_argument("--workers", type=int, default=2, help="Number of concurrent API calls.")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per image after failures.")
    parser.add_argument("--detail", default="low", choices=("low", "high", "auto"), help="Vision detail hint.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum response tokens for each image description.",
    )
    parser.add_argument(
        "--no-response-format",
        action="store_true",
        help="Do not send response_format=json_object for APIs that do not support it.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N images. 0 means no limit.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess files that already exist in the output JSONL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be processed without calling the API.",
    )
    parser.add_argument(
        "--verbose-failures",
        action="store_true",
        help="Print server response details for failed images while still writing them to JSONL.",
    )
    return parser.parse_args()


def iter_images(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


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
                done.add(item["path"])
    return done


def image_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def chat_completion_payload(
    path: Path,
    model: str,
    detail: str,
    max_tokens: int,
    use_response_format: bool,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一个表情包图片标注助手。输出必须简短、准确、适合语义检索。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DEFAULT_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(path),
                            "detail": detail,
                        },
                    },
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    if use_response_format:
        payload["response_format"] = {"type": "json_object"}
    return payload


def response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return dict(response)


def extract_api_error(exc: APIError) -> ApiRequestError:
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None)
    reason = None
    headers: dict[str, str] = {}
    body = None

    if response is not None:
        status = status or getattr(response, "status_code", None)
        reason = getattr(response, "reason_phrase", None) or getattr(response, "reason", None)
        response_headers = getattr(response, "headers", None)
        if response_headers is not None:
            headers = dict(response_headers.items())
        try:
            body = response.text
        except Exception:
            body = None

    if body is None:
        body = str(getattr(exc, "body", "") or "")

    message = f"HTTP {status}: {body}" if status else str(exc)
    return ApiRequestError(message, status=status, reason=reason, headers=headers, body=body)


def create_chat_completion(client: OpenAI, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.chat.completions.create(**payload)
    return response_to_dict(response)


def parse_caption(raw_content: str) -> tuple[dict[str, Any], str]:
    return parse_caption_json(raw_content)


def describe_image(
    path: Path,
    relative_path: str,
    base_url: str,
    api_key: str,
    model: str,
    detail: str,
    max_tokens: int,
    use_response_format: bool,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    last_error = ""
    server_response: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    started_at = time.time()
    endpoint = base_url.rstrip("/") + "/chat/completions"
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
    )
    for attempt in range(1, retries + 2):
        try:
            payload = chat_completion_payload(path, model, detail, max_tokens, use_response_format)
            response = create_chat_completion(client, payload)
            raw_content = response["choices"][0]["message"]["content"]
            caption, semantic_text = parse_caption(raw_content)
            return {
                "path": relative_path,
                "ok": True,
                "model": model,
                "endpoint": endpoint,
                "caption": caption,
                "semantic_text": semantic_text,
                "usage": response.get("usage"),
                "elapsed_seconds": round(time.time() - started_at, 3),
            }
        except APIError as exc:
            api_error = extract_api_error(exc)
            last_error = str(api_error)
            server_response = api_error.to_log()
            attempts.append(
                {
                    "attempt": attempt,
                    "error": last_error,
                    "server_response": server_response,
                }
            )
        except (APIConnectionError, APITimeoutError) as exc:
            api_error = ApiRequestError(str(exc))
            last_error = str(api_error)
            server_response = api_error.to_log()
            attempts.append(
                {
                    "attempt": attempt,
                    "error": last_error,
                    "server_response": server_response,
                }
            )
        except ApiRequestError as exc:
            last_error = str(exc)
            server_response = exc.to_log()
            attempts.append(
                {
                    "attempt": attempt,
                    "error": last_error,
                    "server_response": server_response,
                }
            )
        except (KeyError, json.JSONDecodeError, TimeoutError) as exc:
            last_error = str(exc)
            attempts.append({"attempt": attempt, "error": last_error})

        if attempt <= retries:
            time.sleep(min(30, 2**attempt) + random.random())

    item = {
        "path": relative_path,
        "ok": False,
        "model": model,
        "endpoint": endpoint,
        "error": last_error,
        "attempts": attempts,
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    if server_response is not None:
        item["server_response"] = server_response
    return item


def append_jsonl(output_path: Path, item: dict[str, Any]) -> None:
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def summarize_failure(item: dict[str, Any]) -> str:
    server_response = item.get("server_response") or {}
    status = server_response.get("status")
    reason = server_response.get("reason")
    body = str(server_response.get("body") or item.get("error") or "")
    body = body.replace("\n", "\\n")
    if len(body) > 500:
        body = body[:500] + "...<truncated>"
    parts = [f"path={item.get('path')}", f"error={item.get('error')}"]
    if status is not None:
        parts.append(f"status={status}")
    if reason:
        parts.append(f"reason={reason}")
    if body:
        parts.append(f"body={body}")
    return " | ".join(parts)


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images_dir)
    output_path = Path(args.output)
    endpoint = args.base_url.rstrip("/") + "/chat/completions"

    if not images_dir.exists():
        print(f"Images directory does not exist: {images_dir}", file=sys.stderr)
        return 2

    images = iter_images(images_dir)
    done_paths = set() if args.overwrite else load_done_paths(output_path)
    tasks = [
        path
        for path in images
        if str(path.relative_to(images_dir.parent)) not in done_paths
    ]
    if args.limit:
        tasks = tasks[: args.limit]

    print(f"Found {len(images)} image(s). Pending: {len(tasks)}.")
    print(f"Output: {output_path}")
    print(f"Endpoint: {endpoint}")
    print(f"Model: {args.model}")

    if args.dry_run:
        for path in tasks:
            print(path)
        return 0

    if not args.api_key:
        print("Missing API key. Set OPENAI_API_KEY/API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path(".") else None
    completed = 0
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_path = {
            executor.submit(
                describe_image,
                path,
                str(path.relative_to(images_dir.parent)),
                args.base_url.rstrip("/"),
                args.api_key,
                args.model,
                args.detail,
                args.max_tokens,
                not args.no_response_format,
                args.timeout,
                args.retries,
            ): path
            for path in tasks
        }
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                item = future.result()
            except Exception as exc:  # Defensive: keep long indexing jobs moving.
                item = {
                    "path": str(path.relative_to(images_dir.parent)),
                    "ok": False,
                    "model": args.model,
                    "error": str(exc),
                }
            append_jsonl(output_path, item)
            completed += 1
            failed += 0 if item.get("ok") else 1
            status = "ok" if item.get("ok") else "failed"
            print(f"[{completed}/{len(tasks)}] {status}: {item['path']}", flush=True)
            if args.verbose_failures and not item.get("ok"):
                print("  " + summarize_failure(item), flush=True)

    print(f"Done. Success: {completed - failed}. Failed: {failed}.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
