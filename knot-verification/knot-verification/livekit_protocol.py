"""Shared, testable protocol for the robot and LiveKit operator clients."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


COMMAND_METHOD = "robot.command"
TELEMETRY_TRACK = "robot.telemetry"
VIDEO_TRACK = "robot.camera"
ALLOWED_COMMANDS = {
    "inspect_now",
    "pause_inference",
    "resume_inference",
    "status",
}


@dataclass(frozen=True, slots=True)
class RobotCommand:
    command: str
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class InspectionTelemetry:
    frame_id: int
    label: str | None
    confidence: float | None
    detection_confidence: float | None
    box: tuple[float, float, float, float] | None
    inference_latency_ms: float | None
    inference_paused: bool
    event: str = "inspection"


def parse_command(
    payload: str,
    *,
    caller_identity: str,
    authorized_identity: str,
) -> RobotCommand:
    if caller_identity != authorized_identity:
        raise PermissionError(f"participant {caller_identity!r} is not authorized")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("command payload must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("command payload must be a JSON object")
    unknown = set(value) - {"command", "request_id"}
    if unknown:
        raise ValueError(f"unknown command fields: {sorted(unknown)}")
    command = value.get("command")
    if not isinstance(command, str) or command not in ALLOWED_COMMANDS:
        raise ValueError(f"command must be one of {sorted(ALLOWED_COMMANDS)}")
    request_id = value.get("request_id")
    if request_id is not None and not isinstance(request_id, str):
        raise ValueError("request_id must be a string when provided")
    if isinstance(request_id, str) and len(request_id) > 128:
        raise ValueError("request_id must be at most 128 characters")
    return RobotCommand(command=command, request_id=request_id)


def encode_message(event: str, payload: dict[str, Any]) -> bytes:
    message = {
        "schema_version": 1,
        "event": event,
        "sent_at": datetime.now(UTC).isoformat(),
        **payload,
    }
    return json.dumps(
        message,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def encode_inspection(telemetry: InspectionTelemetry) -> bytes:
    return encode_message(telemetry.event, asdict(telemetry))


def command_payload(command: str, request_id: str | None = None) -> str:
    if command not in ALLOWED_COMMANDS:
        raise ValueError(f"command must be one of {sorted(ALLOWED_COMMANDS)}")
    payload: dict[str, str] = {"command": command}
    if request_id:
        payload["request_id"] = request_id
    return json.dumps(payload, separators=(",", ":"))
