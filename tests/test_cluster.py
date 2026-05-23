from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from emoji_search.cluster import build_features, cluster_features, export_cluster_csv, load_features


class ClusterTests(unittest.TestCase):
    def test_handcrafted_cluster_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images_dir = root / "images"
            images_dir.mkdir()
            Image.new("RGB", (32, 32), (250, 20, 20)).save(images_dir / "red.png")
            Image.new("RGB", (32, 32), (248, 18, 18)).save(images_dir / "red2.png")
            Image.new("RGB", (32, 32), (20, 20, 250)).save(images_dir / "blue.png")

            features_path = root / "features.pkl"
            csv_path = root / "clusters.csv"
            summary = build_features(
                images_dir=images_dir,
                output_path=features_path,
                encoder="handcrafted",
                clip_model="clip-ViT-B-32",
                batch_size=2,
                limit=0,
            )
            self.assertEqual(summary["valid_records"], 3)

            payload = load_features(features_path)
            clustered = cluster_features(payload, distance_threshold=1.5)
            export = export_cluster_csv(clustered["records"], csv_path, payload["encoder"])

            self.assertEqual(export["rows"], 3)
            text = csv_path.read_text(encoding="utf-8")
            self.assertIn("visual_cluster_id", text)
            self.assertIn("near_duplicate_group", text)


if __name__ == "__main__":
    unittest.main()
