from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent
APP_PATH = PROJECT_ROOT / "web" / "app.py"


def load_web_app():
    spec = importlib.util.spec_from_file_location("summit_guardian_web", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load web app from {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class WebDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_web_app()
        cls.client = cls.module.app.test_client()

    def test_mission_control_renders_verified_project_data(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"SUMMIT", response.data)
        self.assertIn(b"Three signals. One confident decision.", response.data)
        self.assertIn(str(len(self.module.ANNOTATIONS)).encode(), response.data)

    def test_demo_status_reports_model_and_livekit_state(self) -> None:
        with patch.object(
            self.module.socket,
            "create_connection",
            side_effect=OSError("offline"),
        ):
            response = self.client.get("/api/demo-status")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["livekit_online"])
        self.assertTrue(payload["avalanche_ready"])
        self.assertEqual(
            payload["edge_inference"],
            "ready" if (
                self.module.config.YOLO_WEIGHTS_OUT.exists()
                and self.module.config.CLASSIFIER_MODEL_OUT.exists()
            ) else "setup_required",
        )

    def test_avalanche_dashboard_uses_chronological_model_artifacts(self) -> None:
        response = self.client.get("/avalanche")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Read the slope", response.data)
        self.assertIn(b"Actual risk vs. model signal", response.data)
        self.assertIn(b"Official avalanche forecasts remain authoritative", response.data)

        api_response = self.client.get("/api/avalanche-summary")
        payload = api_response.get_json()
        self.assertEqual(api_response.status_code, 200)
        self.assertEqual(payload["samples"], 143)
        self.assertEqual(payload["numeric_features"], 773)
        self.assertEqual(len(payload["predictions"]), 29)


if __name__ == "__main__":
    unittest.main()
