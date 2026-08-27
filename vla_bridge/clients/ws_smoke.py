#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import websockets


async def run(url: str, token: str, force_ipv4: bool = False) -> None:
    connect_options = {"open_timeout": 15, "close_timeout": 5}
    if force_ipv4:
        parsed = urlsplit(url)
        if not parsed.hostname:
            raise ValueError("WebSocket URL has no hostname")
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        addresses = socket.getaddrinfo(parsed.hostname, port, socket.AF_INET, socket.SOCK_STREAM)
        if not addresses:
            raise RuntimeError(f"No IPv4 address found for {parsed.hostname}")
        connect_options.update({"host": addresses[0][4][0], "port": port})
        if parsed.scheme == "wss":
            connect_options["server_hostname"] = parsed.hostname

    async with websockets.connect(url, **connect_options) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "hello",
                    "protocol": "vla-bridge.v1",
                    "token": token,
                    "client": "python-smoke-test",
                }
            )
        )
        ready = json.loads(await websocket.recv())
        if ready.get("type") != "ready":
            raise RuntimeError(f"handshake failed: {ready}")

        request_id = "smoke-0001"
        await websocket.send(
            json.dumps(
                {
                    "type": "state",
                    "request_id": request_id,
                    "episode_id": "smoke-episode",
                    "step": 0,
                    "state": {
                        "joint_positions": [0.1, -0.2, 0.3, 0.0, 0.5, 0.0, -0.1],
                        "gripper": 0.0,
                    },
                }
            )
        )
        started = time.perf_counter()
        action = json.loads(await websocket.recv())
        round_trip_ms = round((time.perf_counter() - started) * 1000, 3)
        if action.get("type") != "action" or action.get("request_id") != request_id:
            raise RuntimeError(f"action response mismatch: {action}")
        print(
            json.dumps(
                {"ready": ready, "response": action, "round_trip_ms": round_trip_ms},
                ensure_ascii=False,
                indent=2,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end VLA bridge WebSocket smoke test")
    parser.add_argument("--url", required=True, help="ws:// or wss:// endpoint ending in /v1/ws")
    parser.add_argument("--token", default=os.getenv("VLA_BRIDGE_TOKEN"))
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--force-ipv4", action="store_true")
    args = parser.parse_args()
    if args.token_file:
        args.token = args.token_file.expanduser().read_text(encoding="utf-8").strip()
    if not args.token:
        print("Provide --token, --token-file, or VLA_BRIDGE_TOKEN", file=sys.stderr)
        raise SystemExit(2)
    asyncio.run(run(args.url, args.token, force_ipv4=args.force_ipv4))


if __name__ == "__main__":
    main()
