from __future__ import annotations

import json
import unittest

from livekit_protocol import (
    InspectionTelemetry,
    command_payload,
    encode_inspection,
    parse_command,
)


class CommandTests(unittest.TestCase):
    def test_authorized_command_round_trip(self) -> None:
        payload = command_payload("inspect_now", "abc")
        command = parse_command(
            payload,
            caller_identity="operator",
            authorized_identity="operator",
        )
        self.assertEqual(command.command, "inspect_now")
        self.assertEqual(command.request_id, "abc")

    def test_rejects_unauthorized_participant(self) -> None:
        with self.assertRaises(PermissionError):
            parse_command(
                '{"command":"status"}',
                caller_identity="intruder",
                authorized_identity="operator",
            )

    def test_rejects_unknown_or_extra_command_data(self) -> None:
        with self.assertRaises(ValueError):
            parse_command(
                '{"command":"drive","speed":1}',
                caller_identity="operator",
                authorized_identity="operator",
            )


class TelemetryTests(unittest.TestCase):
    def test_inspection_telemetry_is_strict_json(self) -> None:
        payload = encode_inspection(
            InspectionTelemetry(
                frame_id=7,
                label="Correct",
                confidence=0.9,
                detection_confidence=0.8,
                box=(1.0, 2.0, 3.0, 4.0),
                inference_latency_ms=25.5,
                inference_paused=False,
            )
        )
        decoded = json.loads(payload)
        self.assertEqual(decoded["schema_version"], 1)
        self.assertEqual(decoded["event"], "inspection")
        self.assertEqual(decoded["box"], [1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
