from __future__ import annotations

import unittest

from emoji_search.caption_schema import FIELD_NAMES, normalize_caption, normalize_text, parse_caption_json, semantic_text


class CaptionSchemaTests(unittest.TestCase):
    def test_normalize_caption_fills_missing_fields(self) -> None:
        caption = normalize_caption({"expression": "大哭", "text_in_image": ""})

        self.assertEqual(caption["expression"], "大哭")
        self.assertEqual(caption["text_in_image"], "NONE")
        self.assertEqual(set(caption), set(FIELD_NAMES))

    def test_semantic_text_skips_placeholders(self) -> None:
        text = semantic_text({"expression": "大哭", "action": "NONE", "notes": "流泪"})

        self.assertEqual(text, "大哭；流泪")

    def test_normalize_text_joins_model_lists(self) -> None:
        self.assertEqual(normalize_text(["等暑假", "熬日子"]), "等暑假；熬日子")

    def test_parse_caption_json_normalizes_field_values(self) -> None:
        caption, text = parse_caption_json('{"wechat_keyword": ["等暑假", "熬日子"], "expression": "无奈"}')

        self.assertEqual(caption["wechat_keyword"], "等暑假；熬日子")
        self.assertEqual(caption["expression"], "无奈")
        self.assertIn("等暑假", text)


if __name__ == "__main__":
    unittest.main()
