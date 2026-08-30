"""LiveKit connection and token helpers with environment-only credentials."""

from __future__ import annotations

import os
from pathlib import Path

from livekit import api
from dotenv import load_dotenv


DEFAULT_ENV_FILE = Path(__file__).with_name(".env.livekit")


def load_livekit_env(path: Path | None = None) -> bool:
    return load_dotenv(path or DEFAULT_ENV_FILE, override=False)


def require_url() -> str:
    load_livekit_env()
    url = os.getenv("LIVEKIT_URL")
    if not url:
        raise RuntimeError("LIVEKIT_URL is required")
    if not url.startswith(("ws://", "wss://")):
        raise RuntimeError("LIVEKIT_URL must start with ws:// or wss://")
    return url


def token_from_env(
    *,
    identity: str,
    room_name: str,
    can_publish: bool,
    can_subscribe: bool,
    token_variable: str = "LIVEKIT_TOKEN",
) -> str:
    load_livekit_env()
    supplied_token = os.getenv(token_variable)
    if supplied_token:
        return supplied_token

    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError(
            f"{token_variable}, or LIVEKIT_API_KEY and LIVEKIT_API_SECRET, "
            "must be set"
        )
    return (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=can_publish,
                can_publish_data=True,
                can_subscribe=can_subscribe,
            )
        )
        .to_jwt()
    )
