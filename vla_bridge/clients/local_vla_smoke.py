#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
import websockets


def make_observation(request_id: str) -> dict[str, object]:
    height, width = 240, 320
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = x
    image[..., 1] = y
    image[..., 2] = (x.astype(np.uint16) + y.astype(np.uint16)) // 2
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=90)
    return {
        "type": "state",
        "request_id": request_id,
        "episode_id": "dexvla-smoke-episode",
        "step": 0,
        "instruction": "Use both arms to grasp and hand over the object.",
        "state": np.linspace(-0.25, 0.25, 14, dtype=np.float32).tolist(),
        "images": {
            "front": {
                "encoding": "jpeg_base64",
                "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
        },
    }


def summarize(response: dict[str, object]) -> dict[str, object]:
    chunk = response.get("action_chunk")
    chunk_shape = None
    if isinstance(chunk, list) and chunk and isinstance(chunk[0], list):
        chunk_shape = [len(chunk), len(chunk[0])]
    return {
        key: value
        for key, value in response.items()
        if key != "action_chunk"
    } | {"action_chunk_shape": chunk_shape}


def run_http(url: str, token: str) -> None:
    payload = make_observation("http-dexvla-0001")
    payload.pop("type")
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=300) as response:
        body = json.loads(response.read())
        status = response.status
    if status != 200 or body.get("type") != "action":
        raise RuntimeError(f"HTTP inference failed: status={status}, body={body}")
    print(
        json.dumps(
            {
                "status": status,
                "round_trip_ms": round((time.perf_counter() - started) * 1000, 3),
                "response": summarize(body),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


async def run_ws(url: str, token: str) -> None:
    async with websockets.connect(url, open_timeout=15, close_timeout=5, max_size=16 * 1024 * 1024) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "hello",
                    "protocol": "vla-bridge.v1",
                    "token": token,
                    "client": "dexvla-local-smoke",
                }
            )
        )
        ready = json.loads(await websocket.recv())
        if ready.get("type") != "ready" or not ready.get("model_loaded"):
            raise RuntimeError(f"WebSocket handshake/model readiness failed: {ready}")

        payload = make_observation("ws-dexvla-0001")
        started = time.perf_counter()
        await websocket.send(json.dumps(payload))
        response = json.loads(await websocket.recv())
        if response.get("type") != "action" or response.get("request_id") != payload["request_id"]:
            raise RuntimeError(f"WebSocket inference failed: {response}")
        print(
            json.dumps(
                {
                    "hello": ready,
                    "state": {
                        "request_id": payload["request_id"],
                        "image_shape": [240, 320, 3],
                        "state_shape": [14],
                    },
                    "round_trip_ms": round((time.perf_counter() - started) * 1000, 3),
                    "response": summarize(response),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Local real-DexVLA bridge smoke test")
    parser.add_argument("mode", choices=("http", "ws"))
    parser.add_argument("--url")
    parser.add_argument("--token", default=os.getenv("VLA_BRIDGE_TOKEN"))
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args()
    if args.token_file:
        args.token = args.token_file.expanduser().read_text(encoding="utf-8").strip()
    if not args.token:
        print("Provide --token, --token-file, or VLA_BRIDGE_TOKEN", file=sys.stderr)
        raise SystemExit(2)

    if args.mode == "http":
        run_http(args.url or "http://127.0.0.1:8765/v1/action", args.token)
    else:
        asyncio.run(run_ws(args.url or "ws://127.0.0.1:8765/v1/ws", args.token))


if __name__ == "__main__":
    main()
