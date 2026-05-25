# Project Status

Date: 2026-05-23

## Implemented

- FastAPI backend for image management and semantic search.
- Static browser UI for search, preview, upload, delete, index sync, and model load/unload.
- Vision caption CLI that writes `image_index.jsonl`.
- SentenceTransformer vector index CLI that writes `image_vectors.pkl`.
- Shared caption schema in `emoji_search.caption_schema`.
- Central path/env defaults in `emoji_search.config`.
- Generic image-folder inspection/import CLI in `emoji_search.image_folder`.
- QQ Chat Exporter pack inspection/import CLI in `emoji_search.qq_pack`.
- `.env` loading/writing helpers in `emoji_search.envfile`.
- Interactive API-first workflow CLI in `emoji_search.cli`.
- Human-editable CSV catalog export/apply CLI in `emoji_search.catalog`.
- Local CLIP-based pre-classification and clustering CLI in `emoji_search.cluster`.
- Qwen cluster-representative captioning CLI in `emoji_search.qwen_representatives`.
- Local Raycast extension in `raycast/emoji-search` for desktop search, preview, copy, paste, and catalog/index actions.

## Local Data Finding

The real sticker corpus, QQ container paths, generated manifests, generated captions, and vector indexes are local-only. They are intentionally ignored by Git.

Observed local import shape:

- QQ Chat Exporter metadata may have `pack_info.json` even when `stickers/` is empty.
- QQ's local `personal_emoji/Ori` directory can be used as an additional `--source-dir`.
- The importer matches source files by md5/name and writes the local manifest to `image_manifest.jsonl`.

## Local Import Done

- Imported 500 matched stickers into `images/` using hardlinks.
- Wrote 500 import-manifest rows to `image_manifest.jsonl`.
- Exported local CLIP clustering to `image_clusters.csv`.
- Ran Qwen captioning for the full 500-image corpus.
- Wrote 500 unique caption records to `image_index.jsonl`.
- One image was blocked by the provider's image inspection; it has a local "manual review needed" placeholder instead of a Qwen-generated caption.
- Rebuilt `image_vectors.pkl` with 500 searchable records.
- Exported a 500-row human-editable CSV to `emoji_catalog.csv`.

## Next Tasks

1. Manually review `emoji_catalog.csv`, especially `manual_tags`, `wechat_keyword`, and the provider-blocked placeholder row.
2. Add private aliases for real classmates or local memes in `manual_tags`.
3. Run `uv run python -m emoji_search.catalog apply emoji_catalog.csv --caption-index image_index.jsonl`.
4. Rebuild `image_vectors.pkl`.
5. Open the FastAPI app or Raycast extension and check search quality on common queries.

## Engineering Gaps

- Add API-level tests for `backend.py`.
- Add a small fixture image corpus for repeatable local tests.
- Add search-quality smoke tests with known captions and expected top results.
- Decide whether to move top-level `main.py`, `backend.py`, and `semantic_search.py` fully into the `emoji_search` package.
- Add CI once the project has a remote collaboration workflow.
