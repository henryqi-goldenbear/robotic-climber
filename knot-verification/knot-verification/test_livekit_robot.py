from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any, Callable

from livekit.rtc.rpc import RpcInvocationData

from livekit_protocol import COMMAND_METHOD
from livekit_robot import RobotRuntime, inspection_telemetry, register_command_rpc
from pipeline import InspectionResult


class FakeLocalParticipant:
    def __init__(self) -> None:
        self.handlers: dict[
            str, Callable[[RpcInvocationData], Any]
        ] = {}

    def register_rpc_method(
        self, method: str
    ) -> Callable[
        [Callable[[RpcInvocationData], Any]],
        Callable[[RpcInvocationData], Any],
    ]:
        def register(
            handler: Callable[[RpcInvocationData], Any]
        ) -> Callable[[RpcInvocationData], Any]:
            self.handlers[method] = handler
            return handler

        return register


class FakeRoom:
    def __init__(self) -> None:
        self.local_participant = FakeLocalParticipant()


class RpcHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_authorized_handler_updates_inference_state(self) -> None:
        room = FakeRoom()
        runtime = RobotRuntime()
        register_command_rpc(room, runtime, "operator")
        handler = room.local_participant.handlers[COMMAND_METHOD]

        response = await handler(
            SimpleNamespace(
                payload='{"command":"pause_inference","request_id":"abc"}',
                caller_identity="operator",
            )
        )

        decoded = json.loads(response)
        self.assertTrue(decoded["ok"])
        self.assertTrue(runtime.inference_paused)
        self.assertEqual(decoded["request_id"], "abc")

    async def test_handler_rejects_another_participant(self) -> None:
        room = FakeRoom()
        runtime = RobotRuntime()
        register_command_rpc(room, runtime, "operator")
        handler = room.local_participant.handlers[COMMAND_METHOD]

        with self.assertRaises(PermissionError):
            await handler(
                SimpleNamespace(
                    payload='{"command":"status"}',
                    caller_identity="intruder",
                )
            )


class TelemetryAlignmentTests(unittest.TestCase):
    def test_telemetry_uses_the_inferred_frame_id(self) -> None:
        runtime = RobotRuntime(frame_id=12, last_inference_frame_id=7)
        result = InspectionResult(
            label="Correct",
            confidence=0.9,
            detection_confidence=0.8,
            box=(1.0, 2.0, 3.0, 4.0),
        )

        telemetry = inspection_telemetry(runtime, result, inference_frame_id=7)

        self.assertEqual(telemetry.frame_id, 7)


if __name__ == "__main__":
    unittest.main()
