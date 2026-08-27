"""Synchronous client for the 4080 VLA bridge.

WebSocket is preferred. In ``auto`` mode, inference falls back to HTTP if the
WebSocket path fails. Authentication supports bearer/query/hello-token modes.
"""

from __future__ import annotations

import json
import ssl
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests


@dataclass(frozen=True)
class RemoteReply:
    payload: dict[str, Any]
    transport: str


class RemoteVLAClient:
    def __init__(
        self,
        *,
        ws_url: str,
        http_url: str,
        health_url: str,
        token: str,
        transport: str = "auto",
        ws_auth_mode: str = "auto",
        timeout_s: float = 5.0,
        client_name: str = "simulation-web",
    ) -> None:
        self.ws_url = ws_url
        self.http_url = http_url
        self.health_url = health_url
        self.token = token.strip()
        self.transport = transport
        self.ws_auth_mode = ws_auth_mode
        self.timeout_s = float(timeout_s)
        self.client_name = client_name
        self._ws: Any = None
        self._lock = threading.Lock()
        self._active_ws_auth_mode: str | None = None

    @staticmethod
    def load_token(env_value: str, token_file: Path) -> str:
        if env_value.strip():
            return env_value.strip()
        try:
            return token_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return ""

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _url_with_token(self, url: str) -> str:
        if not self.token:
            return url
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query.setdefault("token", self.token)
        return urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
        )

    def _ws_modes(self) -> tuple[str, ...]:
        if self.ws_auth_mode == "auto":
            return ("bearer", "query", "hello") if self.token else ("none",)
        return (self.ws_auth_mode,)

    def _connect_ws(self) -> None:
        try:
            import websocket
        except ImportError as exc:
            raise RuntimeError("WebSocket transport requires websocket-client") from exc

        last_error: BaseException | None = None
        for mode in self._ws_modes():
            headers: list[str] = []
            url = self.ws_url
            if mode == "bearer" and self.token:
                headers.append(f"Authorization: Bearer {self.token}")
            elif mode == "query":
                url = self._url_with_token(url)

            ws = None
            try:
                ws = websocket.create_connection(
                    url,
                    timeout=self.timeout_s,
                    header=headers,
                    sslopt={"cert_reqs": ssl.CERT_REQUIRED},
                )
                hello: dict[str, Any] = {
                    "type": "hello",
                    "client": self.client_name,
                }
                if mode == "hello" and self.token:
                    hello["token"] = self.token
                ws.send(json.dumps(hello, ensure_ascii=False))
                self._ws = ws
                self._active_ws_auth_mode = mode
                return
            except BaseException as exc:
                last_error = exc
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass
        raise RuntimeError(f"Unable to connect to remote WebSocket: {last_error!r}")

    def connect(self) -> str:
        """Proactively establish the preferred transport before robot motion."""
        with self._lock:
            if self.transport == "http":
                return "http"
            if self._ws is not None:
                return "ws"
            try:
                self._connect_ws()
                return "ws"
            except Exception:
                self._close_ws()
                if self.transport == "ws":
                    raise
                return "http"

    def _close_ws(self) -> None:
        ws, self._ws = self._ws, None
        self._active_ws_auth_mode = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def _infer_ws(self, payload: dict[str, Any]) -> RemoteReply:
        if self._ws is None:
            self._connect_ws()
        assert self._ws is not None
        request_id = str(payload.get("request_id", ""))
        self._ws.send(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        while True:
            raw = self._ws.recv()
            if raw is None:
                raise RuntimeError("remote WebSocket closed")
            message = json.loads(raw)
            if not isinstance(message, dict):
                continue
            kind = str(message.get("type", ""))
            if kind in {"hello", "ready", "ack", "pong"}:
                continue
            if request_id and message.get("request_id") not in {None, request_id}:
                continue
            if kind == "error":
                raise RuntimeError(
                    f"remote inference error: {message.get('code')} "
                    f"{message.get('message')}"
                )
            if (
                kind == "action"
                or "action" in message
                or "action_chunk" in message
                or "command" in message
            ):
                return RemoteReply(message, "ws")

    def _infer_http(self, payload: dict[str, Any]) -> RemoteReply:
        response = requests.post(
            self.http_url,
            json=payload,
            headers=self._auth_headers(),
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        message = response.json()
        if not isinstance(message, dict):
            raise RuntimeError("HTTP inference response is not a JSON object")
        if message.get("type") == "error":
            raise RuntimeError(
                f"remote inference error: {message.get('code')} {message.get('message')}"
            )
        return RemoteReply(message, "http")

    def infer(self, payload: dict[str, Any]) -> RemoteReply:
        with self._lock:
            if self.transport == "http":
                return self._infer_http(payload)
            try:
                return self._infer_ws(payload)
            except Exception:
                self._close_ws()
                if self.transport == "ws":
                    raise
                return self._infer_http(payload)

    def health(self) -> dict[str, Any]:
        response = requests.get(
            self.health_url,
            headers=self._auth_headers(),
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        value = response.json()
        return value if isinstance(value, dict) else {"value": value}

    def close(self) -> None:
        with self._lock:
            self._close_ws()
