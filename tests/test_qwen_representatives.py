from __future__ import annotations

import unittest

from emoji_search.qwen_representatives import select_representatives


class QwenRepresentativesTests(unittest.TestCase):
    def test_select_representatives_prefers_large_clusters_and_skips_done(self) -> None:
        rows = [
            {
                "path": "images/a.png",
                "visual_cluster_id": "visual_0001",
                "cluster_size": "2",
                "asset_type": "cartoon_character",
            },
            {
                "path": "images/b.png",
                "visual_cluster_id": "visual_0002",
                "cluster_size": "5",
                "asset_type": "real_person",
            },
            {
                "path": "images/c.png",
                "visual_cluster_id": "visual_0003",
                "cluster_size": "1",
                "asset_type": "text_meme",
            },
        ]

        selected = select_representatives(
            rows,
            done_paths={"images/b.png"},
            overwrite=False,
            min_cluster_size=2,
            asset_types=set(),
            limit=0,
        )

        self.assertEqual([row["path"] for row in selected], ["images/a.png"])

    def test_select_representatives_filters_asset_type(self) -> None:
        rows = [
            {
                "path": "images/a.png",
                "visual_cluster_id": "visual_0001",
                "cluster_size": "2",
                "asset_type": "cartoon_character",
            },
            {
                "path": "images/b.png",
                "visual_cluster_id": "visual_0002",
                "cluster_size": "2",
                "asset_type": "real_person",
            },
        ]

        selected = select_representatives(
            rows,
            done_paths=set(),
            overwrite=False,
            min_cluster_size=1,
            asset_types={"real_person"},
            limit=0,
        )

        self.assertEqual([row["path"] for row in selected], ["images/b.png"])


if __name__ == "__main__":
    unittest.main()
