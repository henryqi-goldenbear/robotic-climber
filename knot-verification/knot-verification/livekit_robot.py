"""Publish onboard knot inspection video and telemetry over LiveKit."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, Sequence

import cv2
import numpy as np
from livekit import rtc
from livekit.rtc.rpc import RpcInvocationData
from PIL import Image

from livekit_connection import require_url, token_from_env
from livekit_protocol import (
    COMMAND_METHOD,
    TELEMETRY_TRACK,
    VIDEO_TRACK,
    InspectionTelemetry,
    encode_inspection,
    encode_message,
    parse_command,
)
from pipeline import InspectionResult, KnotInspectionPipeline


LOGGER = logging.getLogger("livekit_robot")
RpcHandler = Callable[[RpcInvocationData], Awaitable[str]]


class RpcRegistrar(Protocol):
    def register_rpc_method(
        self, method_name: str
    ) -> Callable[[RpcHandler], RpcHandler]: ...


class RpcRoom(Protocol):
    @property
    def local_participant(self) -> RpcRegistrar: ...


@dataclass(slots=True)
class RobotRuntime:
    inference_paused: bool = False
    force_inspection: asyncio.Event = field(default_factory=asyncio.Event)
    last_result: InspectionResult | None = None
    last_latency_ms: float | None = None
    last_inference_frame_id: int | None = None
    frame_id: int = 0
    published_frames: int = 0
    started_at: float = field(default_factory=time.monotonic)

    def status(self) -> dict[str, Any]:
        result = self.last_result
        return {
            "inference_paused": self.inference_paused,
            "frame_id": self.frame_id,
            "published_frames": self.published_frames,
            "uptime_seconds": round(time.monotonic() - self.started_at, 3),
            "last_label": result.label if result else None,
            "last_confidence": result.confidence if result else None,
            "last_inference_latency_ms": self.last_latency_ms,
            "last_inference_frame_id": self.last_inference_frame_id,
        }


def annotate_frame(
    frame_bgr: np.ndarray, result: InspectionResult | None
) -> np.ndarray:
    annotated = frame_bgr.copy()
    if result is None or result.box is None:
        return annotated
    x1, y1, x2, y2 = map(int, result.box)
    color = (0, 200, 0) if result.label == "Correct" else (0, 0, 220)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    confidence = f" {result.confidence:.0%}" if result.confidence is not None else ""
    cv2.putText(
        annotated,
        f"{result.label or 'Unknown'}{confidence}",
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    return annotated


def inspection_telemetry(
    runtime: RobotRuntime,
    result: InspectionResult | None,
    inference_frame_id: int,
) -> InspectionTelemetry:
    return InspectionTelemetry(
        frame_id=inference_frame_id,
        label=result.label if result else None,
        confidence=result.confidence if result else None,
        detection_confidence=result.detection_confidence if result else None,
        box=result.box if result else None,
        inference_latency_ms=runtime.last_latency_ms,
        inference_paused=runtime.inference_paused,
    )


def open_camera(args: argparse.Namespace) -> tuple[cv2.VideoCapture, int, int]:
    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"could not open camera index {args.camera_index}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise RuntimeError(f"could not read camera index {args.camera_index}")
    height, width = frame.shape[:2]
    if width % 2 or height % 2:
        capture.release()
        raise RuntimeError("camera width and height must be even for I420 video")
    return capture, width, height


def push_telemetry(track: rtc.LocalDataTrack, payload: bytes) -> None:
    try:
        track.try_push(
            rtc.DataTrackFrame(
                payload=payload,
                user_timestamp=int(time.time() * 1_000),
            )
        )
    except rtc.PushFrameError as exc:
        LOGGER.warning("telemetry frame dropped: %s", exc)


async def inference_loop(
    pipeline: KnotInspectionPipeline,
    queue: asyncio.Queue[tuple[int, np.ndarray]],
    runtime: RobotRuntime,
    telemetry_track: rtc.LocalDataTrack,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        try:
            frame_id, frame_bgr = await asyncio.wait_for(queue.get(), timeout=0.5)
        except TimeoutError:
            continue
        force = runtime.force_inspection.is_set()
        if runtime.inference_paused and not force:
            continue
        runtime.force_inspection.clear()
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        started = time.perf_counter()
        result = await asyncio.to_thread(pipeline.predict, image)
        runtime.last_latency_ms = (time.perf_counter() - started) * 1_000
        runtime.last_result = result
        runtime.last_inference_frame_id = frame_id
        push_telemetry(
            telemetry_track,
            encode_inspection(
                inspection_telemetry(runtime, result, frame_id)
            ),
        )


async def heartbeat_loop(
    runtime: RobotRuntime,
    telemetry_track: rtc.LocalDataTrack,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        push_telemetry(
            telemetry_track,
            encode_message("heartbeat", runtime.status()),
        )
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=2.0)
        except TimeoutError:
            pass


async def video_loop(
    args: argparse.Namespace,
    capture: cv2.VideoCapture,
    width: int,
    height: int,
    source: rtc.VideoSource,
    inference_queue: asyncio.Queue[tuple[int, np.ndarray]],
    runtime: RobotRuntime,
    stop_event: asyncio.Event,
) -> None:
    interval = 1.0 / args.fps
    next_frame_at = time.perf_counter()
    started_at_ns = time.perf_counter_ns()
    while not stop_event.is_set():
        ok, frame_bgr = await asyncio.to_thread(capture.read)
        if not ok or frame_bgr is None:
            raise RuntimeError("camera frame read failed")
        if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
            frame_bgr = cv2.resize(
                frame_bgr, (width, height), interpolation=cv2.INTER_AREA
            )

        runtime.frame_id += 1
        should_infer = (
            runtime.force_inspection.is_set()
            or runtime.frame_id % args.infer_every == 0
        )
        if should_infer and (not runtime.inference_paused or runtime.force_inspection.is_set()):
            if inference_queue.full():
                try:
                    inference_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            inference_queue.put_nowait((runtime.frame_id, frame_bgr.copy()))

        annotated = annotate_frame(frame_bgr, runtime.last_result)
        i420 = cv2.cvtColor(annotated, cv2.COLOR_BGR2YUV_I420)
        frame = rtc.VideoFrame(
            width,
            height,
            rtc.VideoBufferType.I420,
            np.ascontiguousarray(i420).tobytes(),
        )
        metadata = rtc.FrameMetadata(
            user_timestamp=time.time_ns() // 1_000,
            frame_id=runtime.frame_id & 0xFFFFFFFF,
        )
        source.capture_frame(
            frame,
            timestamp_us=(time.perf_counter_ns() - started_at_ns) // 1_000,
            metadata=metadata,
        )
        runtime.published_frames += 1

        next_frame_at += interval
        delay = next_frame_at - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        else:
            next_frame_at = time.perf_counter()


def register_command_rpc(
    room: RpcRoom,
    runtime: RobotRuntime,
    authorized_operator: str,
) -> None:
    @room.local_participant.register_rpc_method(COMMAND_METHOD)
    async def handle_command(invocation: RpcInvocationData) -> str:
        command = parse_command(
            invocation.payload,
            caller_identity=invocation.caller_identity,
            authorized_identity=authorized_operator,
        )
        if command.command == "pause_inference":
            runtime.inference_paused = True
        elif command.command == "resume_inference":
            runtime.inference_paused = False
        elif command.command == "inspect_now":
            runtime.force_inspection.set()
        response = {
            "ok": True,
            "command": command.command,
            "request_id": command.request_id,
            "status": runtime.status(),
        }
        return json.dumps(response, separators=(",", ":"), allow_nan=False)


async def run(args: argparse.Namespace, stop_event: asyncio.Event) -> None:
    if args.fps <= 0 or args.infer_every <= 0 or args.max_bitrate <= 0:
        raise ValueError("fps, infer-every, and max-bitrate must be positive")
    url = require_url()
    token = token_from_env(
        identity=args.identity,
        room_name=args.room,
        can_publish=True,
        can_subscribe=False,
    )
    capture, width, height = open_camera(args)
    pipeline = KnotInspectionPipeline()
    runtime = RobotRuntime()
    room = rtc.Room()
    source: rtc.VideoSource | None = None
    tasks: list[asyncio.Task[None]] = []

    try:
        await room.connect(url, token)
        register_command_rpc(room, runtime, args.operator_identity)
        source = rtc.VideoSource(width, height)
        video_track = rtc.LocalVideoTrack.create_video_track(VIDEO_TRACK, source)
        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            video_encoding=rtc.VideoEncoding(
                max_framerate=args.fps,
                max_bitrate=args.max_bitrate,
            ),
            frame_metadata_features=[
                rtc.FrameMetadataFeature.FMF_USER_TIMESTAMP,
                rtc.FrameMetadataFeature.FMF_FRAME_ID,
            ],
        )
        await room.local_participant.publish_track(video_track, options)
        telemetry_track = await room.local_participant.publish_data_track(
            name=TELEMETRY_TRACK
        )
        LOGGER.info(
            "connected to %s as %s; operator=%s, camera=%sx%s",
            args.room,
            args.identity,
            args.operator_identity,
            width,
            height,
        )

        inference_queue: asyncio.Queue[tuple[int, np.ndarray]] = asyncio.Queue(
            maxsize=1
        )
        tasks = [
            asyncio.create_task(
                video_loop(
                    args,
                    capture,
                    width,
                    height,
                    source,
                    inference_queue,
                    runtime,
                    stop_event,
                )
            ),
            asyncio.create_task(
                inference_loop(
                    pipeline,
                    inference_queue,
                    runtime,
                    telemetry_track,
                    stop_event,
                )
            ),
            asyncio.create_task(
                heartbeat_loop(runtime, telemetry_track, stop_event)
            ),
        ]
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        stop_event.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        capture.release()
        if source is not None:
            await source.aclose()
        await room.disconnect()


def install_signal_handlers(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except (NotImplementedError, RuntimeError):
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", default=os.getenv("LIVEKIT_ROOM", "himalaya"))
    parser.add_argument(
        "--identity", default=os.getenv("LIVEKIT_ROBOT_IDENTITY", "robot")
    )
    parser.add_argument(
        "--operator-identity",
        default=os.getenv("LIVEKIT_OPERATOR_IDENTITY", "operator"),
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--infer-every", type=int, default=5)
    parser.add_argument("--max-bitrate", type=int, default=1_500_000)
    return parser


async def async_main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    stop_event = asyncio.Event()
    install_signal_handlers(stop_event)
    await run(args, stop_event)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(async_main(argv))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
