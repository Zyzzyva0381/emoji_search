# WeChat Sticker Readiness

This project is primarily a local search/indexing tool. It can help prepare WeChat sticker candidates, but it does not currently publish or upload stickers.

## Metadata

The caption schema includes:

- `wechat_keyword`: short meaning-word candidate for WeChat.
- `usage_context`: practical chat scenario.
- `manual_tags`: private tags and memes that should also be searchable.

After editing these in `emoji_catalog.csv`, run:

```bash
uv run python -m emoji_search.catalog apply emoji_catalog.csv --caption-index image_index.jsonl
uv run python semantic_search.py build --input image_index.jsonl --output image_vectors.pkl
```

## Future Exporter

A later `emoji_search.wechat_export` should:

- select 8/16/24 related images or single-item candidates;
- render main images to a 240x240 canvas;
- preserve GIF animation where useful;
- generate thumbnails;
- write a CSV of `path,wechat_keyword,name,notes`;
- report files that exceed size limits.
