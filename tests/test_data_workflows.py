from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from emoji_search.annotation import export_batches, merge_external_captions, validate_external_captions
from emoji_search.catalog import apply_catalog, export_catalog
from emoji_search.image_folder import import_folder, inspect_folder
from emoji_search.qq_pack import import_pack, inspect_pack


class DataWorkflowTests(unittest.TestCase):
    def test_qq_pack_inspect_and_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_dir = root / "pack"
            stickers_dir = pack_dir / "stickers"
            images_dir = root / "images"
            stickers_dir.mkdir(parents=True)
            source = stickers_dir / "ABCDEF.jpg"
            source.write_bytes(b"fake-image")
            (pack_dir / "pack_info.json").write_text(
                json.dumps(
                    {
                        "packName": "收藏的表情",
                        "stickerCount": 1,
                        "stickers": [
                            {
                                "stickerId": "7",
                                "name": "测试",
                                "path": str(source),
                                "md5": "ABCDEF",
                                "downloaded": True,
                                "fileSize": 10,
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            summary = inspect_pack(pack_dir)
            self.assertEqual(summary["available_source_files"], 1)

            result = import_pack(
                pack_dir=pack_dir,
                source_dirs=[],
                images_dir=images_dir,
                metadata_output=root / "image_manifest.jsonl",
                mode="copy",
                overwrite=False,
                dry_run=False,
                limit=0,
            )

            self.assertEqual(result["imported_or_planned"], 1)
            self.assertEqual(len(list(images_dir.iterdir())), 1)
            self.assertTrue((root / "image_manifest.jsonl").exists())

    def test_annotation_export_validate_and_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            output_dir = root / "annotation"
            images_dir.mkdir()
            (images_dir / "a.png").write_bytes(b"fake-image")

            exported = export_batches(
                images_dir=images_dir,
                output_dir=output_dir,
                caption_index=root / "image_index.jsonl",
                batch_size=50,
                limit=0,
                include_captioned=False,
            )
            self.assertEqual(exported["tasks"], 1)
            self.assertTrue((output_dir / "PROMPT.md").exists())
            self.assertTrue((output_dir / "batch_000.jsonl").exists())

            captions = output_dir / "captions.jsonl"
            captions.write_text(
                json.dumps(
                    {
                        "path": "images/a.png",
                        "caption": {
                            "image_composition": "白底小猫",
                            "character_name": "猫",
                            "expression": "开心",
                            "action": "挥手",
                            "subjective_emotion": "友好",
                            "text_in_image": "NONE",
                            "notes": "问候",
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            validation = validate_external_captions(captions, source_model="test-model")
            self.assertEqual(validation["errors"], [])

            merged = merge_external_captions(
                input_path=captions,
                output_path=root / "image_index.jsonl",
                source_model="test-model",
                overwrite=False,
            )
            self.assertEqual(merged["merged"], 1)
            line = (root / "image_index.jsonl").read_text(encoding="utf-8").strip()
            self.assertIn("白底小猫", line)

    def test_generic_folder_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            images_dir = root / "images"
            source_dir.mkdir()
            (source_dir / "hello world.png").write_bytes(b"same")
            (source_dir / "duplicate.png").write_bytes(b"same")

            inspected = inspect_folder(source_dir)
            self.assertEqual(inspected["image_files"], 2)
            self.assertEqual(inspected["duplicate_files"], 1)

            imported = import_folder(
                source_dir=source_dir,
                images_dir=images_dir,
                metadata_output=root / "image_manifest.jsonl",
                mode="copy",
                overwrite=False,
                dry_run=False,
                keep_duplicates=False,
                prefix="test",
                collection="fixtures",
                limit=0,
            )
            self.assertEqual(imported["imported_or_planned"], 1)
            self.assertEqual(imported["skipped_duplicate"], 1)
            self.assertEqual(len(list(images_dir.iterdir())), 1)

    def test_catalog_export_and_apply_manual_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir()
            (images_dir / "a.png").write_bytes(b"fake-image")
            caption_index = root / "image_index.jsonl"
            manifest = root / "image_manifest.jsonl"
            catalog = root / "emoji_catalog.csv"

            export_summary = export_catalog(
                images_dir=images_dir,
                caption_index=caption_index,
                manifest_path=manifest,
                cluster_path=root / "image_clusters.csv",
                output_path=catalog,
            )
            self.assertEqual(export_summary["rows"], 1)

            text = catalog.read_text(encoding="utf-8")
            text = text.replace("images/a.png,a.png,", "images/a.png,a.png,")
            lines = text.splitlines()
            header = lines[0].split(",")
            row = lines[1].split(",")
            row[header.index("manual_tags")] = "私房梗;常用"
            row[header.index("wechat_keyword")] = "懂了"
            catalog.write_text(",".join(header) + "\n" + ",".join(row) + "\n", encoding="utf-8")

            applied = apply_catalog(input_path=catalog, caption_index=caption_index, overwrite=False)
            self.assertEqual(applied["created"], 1)
            output = caption_index.read_text(encoding="utf-8")
            self.assertIn("私房梗", output)
            self.assertIn("懂了", output)


if __name__ == "__main__":
    unittest.main()
