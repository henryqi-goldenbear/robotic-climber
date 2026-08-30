from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from livekit_connection import load_livekit_env, require_url


class LiveKitEnvironmentTests(unittest.TestCase):
    def test_loads_explicit_environment_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory, ".env.livekit")
            env_file.write_text(
                "LIVEKIT_URL=ws://127.0.0.1:7880\n",
                encoding="ascii",
            )
            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(load_livekit_env(env_file))
                self.assertEqual(require_url(), "ws://127.0.0.1:7880")

    def test_environment_file_does_not_override_process_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory, ".env.livekit")
            env_file.write_text(
                "LIVEKIT_URL=ws://127.0.0.1:7880\n",
                encoding="ascii",
            )
            with patch.dict(
                os.environ,
                {"LIVEKIT_URL": "wss://example.livekit.cloud"},
                clear=True,
            ):
                load_livekit_env(env_file)
                self.assertEqual(require_url(), "wss://example.livekit.cloud")


if __name__ == "__main__":
    unittest.main()
