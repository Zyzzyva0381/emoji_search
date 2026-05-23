from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from emoji_search.envfile import api_settings_from_env, read_env_file, update_env_file


class EnvfileTests(unittest.TestCase):
    def test_read_env_file_strips_inline_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "EMOJI_API_BASE_URL=https://example.test/v1\n"
                "EMOJI_API_MODEL=model-name # comment\n"
                "EMOJI_API_KEY='secret value'\n",
                encoding="utf-8",
            )

            values = read_env_file(env_path)

            self.assertEqual(values["EMOJI_API_BASE_URL"], "https://example.test/v1")
            self.assertEqual(values["EMOJI_API_MODEL"], "model-name")
            self.assertEqual(values["EMOJI_API_KEY"], "secret value")

    def test_api_settings_support_provider_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "BAYES_API_KEY=token\n"
                "CHINAMOBILE_MAAS_BASE_URL=https://provider.test/v1\n"
                "CHINAMOBILE_MAAS_MODEL=vision-model\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = api_settings_from_env(env_path)

            self.assertEqual(settings["api_key"], "token")
            self.assertEqual(settings["base_url"], "https://provider.test/v1")
            self.assertEqual(settings["model"], "vision-model")

    def test_update_env_file_preserves_unrelated_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("# local\nOTHER=value\nEMOJI_API_MODEL=old\n", encoding="utf-8")

            update_env_file(env_path, {"EMOJI_API_MODEL": "new", "EMOJI_API_KEY": "secret"})

            text = env_path.read_text(encoding="utf-8")
            self.assertIn("# local", text)
            self.assertIn("OTHER=value", text)
            self.assertIn("EMOJI_API_MODEL=new", text)
            self.assertIn("EMOJI_API_KEY=secret", text)


if __name__ == "__main__":
    unittest.main()
