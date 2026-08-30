"""Monitor robot telemetry and send an authenticated LiveKit RPC command."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Sequence

import numpy as np
from livekit import rtc
from PIL import Image

from livekit_connection import require_url, token_from_env
from livekit_protocol import (
    ALLOWED_COMMANDS,
    COMMAND_METHOD,
    TELEMETRY_TRACK,
    VIDEO_TRACK,
    command_payload,
)


LOGGER = logging.getLogger("livekit_operator")


async def read_telemetry(track: rtc.RemoteDataTrack) -> None:
    try:
        async for frame in track.subscribe(buffer_size=1):
            try:
                message = json.loads(frame.payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                LOGGER.warning("invalid telemetry frame: %s", exc)
                continue
            print(json.dumps(message, ensure_ascii=False), flush=True)
    except rtc.SubscribeDataTrackError as exc:
        LOGGER.error("telemetry subscription failed: %s", exc.message)


async def save_video_snapshot(
    track: rtc.Track,
    destination: Path,
    snapshot_saved: asyncio.Event,
) -> None:
    stream = rtc.VideoStream.from_track(
        track=track,
        format=rtc.VideoBufferType.RGB24,
        capacity=1,
    )
    try:
        async for event in stream:
            frame = event.frame
            image = np.frombuffer(frame.data, dtype=np.uint8).reshape(
                frame.height, frame.width, 3
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(destination)
            LOGGER.info("saved robot video snapshot to %s", destination)
            snapshot_saved.set()
            return
    finally:
        await stream.aclose()


async def wait_for_robot(
    room: rtc.Room, robot_identity: str, timeout: float
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while robot_identity not in room.remote_participants:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(
                f"robot participant {robot_identity!r} did not join within {timeout}s"
            )
        await asyncio.sleep(0.1)


async def run(args: argparse.Namespace) -> None:
    url = require_url()
    token = token_from_env(
        identity=args.identity,
        room_name=args.room,
        can_publish=False,
        can_subscribe=True,
        token_variable="LIVEKIT_OPERATOR_TOKEN",
    )
    room = rtc.Room()
    tasks: set[asyncio.Task[None]] = set()
    snapshot_saved = asyncio.Event()

    def keep_task(task: asyncio.Task[None]) -> None:
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    @room.on("data_track_published")
    def on_data_track_published(track: rtc.RemoteDataTrack) -> None:
        if (
            track.publisher_identity == args.robot_identity
            and track.info.name == TELEMETRY_TRACK
        ):
            keep_task(asyncio.create_task(read_telemetry(track)))

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if (
            args.snapshot
            and participant.identity == args.robot_identity
            and publication.name == VIDEO_TRACK
            and track.kind == rtc.TrackKind.KIND_VIDEO
            and not snapshot_saved.is_set()
        ):
            keep_task(
                asyncio.create_task(
                    save_video_snapshot(track, args.snapshot, snapshot_saved)
                )
            )

    try:
        await room.connect(url, token)
        await wait_for_robot(room, args.robot_identity, args.connect_timeout)
        LOGGER.info("connected to %s; robot participant is online", args.room)
        if args.command:
            response = await room.local_participant.perform_rpc(
                destination_identity=args.robot_identity,
                method=COMMAND_METHOD,
                payload=command_payload(args.command, str(uuid.uuid4())),
                response_timeout=args.rpc_timeout,
            )
            print(json.dumps(json.loads(response), indent=2), flush=True)
        if args.snapshot:
            await asyncio.wait_for(
                snapshot_saved.wait(), timeout=args.snapshot_timeout
            )
        if args.monitor_seconds > 0:
            await asyncio.sleep(args.monitor_seconds)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await room.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", default=os.getenv("LIVEKIT_ROOM", "himalaya"))
    parser.add_argument(
        "--identity", default=os.getenv("LIVEKIT_OPERATOR_IDENTITY", "operator")
    )
    parser.add_argument(
        "--robot-identity",
        default=os.getenv("LIVEKIT_ROBOT_IDENTITY", "robot"),
    )
    parser.add_argument("--command", choices=sorted(ALLOWED_COMMANDS))
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--monitor-seconds", type=float, default=10.0)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--rpc-timeout", type=float, default=5.0)
    parser.add_argument("--snapshot-timeout", type=float, default=15.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    if min(
        args.monitor_seconds,
        args.connect_timeout,
        args.rpc_timeout,
        args.snapshot_timeout,
    ) < 0:
        raise ValueError("timeouts and monitor-seconds must be non-negative")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
