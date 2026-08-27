from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import PROTOCOL_VERSION, load_settings
from .policies import (
    ExpertLookupPolicy,
    ImageInput,
    PolicyInputError,
    PolicyNotReadyError,
    PolicyRequest,
    RaboVLAPolicy,
    UnavailablePolicy,
    ZeroPolicy,
)

logger = logging.getLogger("uvicorn.error")
settings = load_settings()


class HelloMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["hello"]
    protocol: str | None = Field(default=None, max_length=128)
    token: str | None = Field(default=None, min_length=1, max_length=512)
    client: str | None = Field(default=None, max_length=128)


class ImageMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    encoding: Literal["jpeg_base64"]
    data: str = Field(min_length=1)


class StateMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["state"]
    protocol: str | None = Field(default=None, max_length=128)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    instruction: str | None = Field(default=None, max_length=4096)
    state: Any = None
    full_state: Any = None
    images: dict[str, ImageMessage] | None = None
    episode_id: str | None = Field(default=None, max_length=128)
    step: int | None = Field(default=None, ge=0)
    timestamp_ms: int | None = Field(default=None, ge=0)


class ResetMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["reset"]
    request_id: str = Field(min_length=1, max_length=128)
    episode_id: str | None = Field(default=None, max_length=128)


class PingMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["ping"]
    request_id: str | None = Field(default=None, max_length=128)


class HttpActionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal["state"] | None = None
    protocol: str | None = Field(default=None, max_length=128)
    request_id: str | None = Field(default=None, min_length=1, max_length=128)
    instruction: str | None = Field(default=None, max_length=4096)
    state: Any = None
    full_state: Any = None
    images: dict[str, ImageMessage] | None = None
    episode_id: str | None = Field(default=None, max_length=128)
    step: int | None = Field(default=None, ge=0)
    timestamp_ms: int | None = Field(default=None, ge=0)


async def _load_policy():
    if settings.policy == "zero":
        policy = ZeroPolicy(settings.action_dim)
        await policy.warmup()
        return policy
    if settings.policy == "dexvla":
        from .policies.dexvla_policy import DexVLAPolicy

        policy = await asyncio.to_thread(DexVLAPolicy, settings)
        await policy.warmup()
        return policy
    if settings.policy == "expert_lookup":
        return ExpertLookupPolicy(settings.expert_program_path)
    if settings.policy == "rabo_vla":
        return RaboVLAPolicy(settings.expert_program_path)
    if settings.policy == "joint_vla":
        from .policies.joint_vla_policy import JointVLAPolicy

        return JointVLAPolicy(
            settings.joint_reference_path,
            initial_search=settings.joint_initial_search,
            forward_window=settings.joint_forward_window,
        )
    if settings.policy == "bc_joint_vla":
        from .policies.bc_joint_policy import BCJointVLAPolicy

        return BCJointVLAPolicy(settings.bc_joint_model_dir)
    if settings.policy == "bc_vla":
        # Keep torch/Pillow completely unloaded unless BC is explicitly selected.
        from .policies.bc_vla_policy import BCVLAPolicy

        return await asyncio.to_thread(
            BCVLAPolicy,
            settings.bc_model_dir,
            settings.expert_program_path,
            shadow_only=settings.bc_shadow_only,
            device=settings.bc_device,
            guard_max_advance=settings.bc_guard_max_advance,
        )
    raise ValueError(f"Unsupported VLA_POLICY: {settings.policy}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.policy_lock = asyncio.Lock()
    try:
        app.state.policy = await _load_policy()
    except Exception as exc:
        logger.exception("Policy startup failed: %s", settings.policy)
        app.state.policy = UnavailablePolicy(
            name=settings.policy,
            error=str(exc),
            output_action_dim=7 if settings.output_mode == "compat7" else 0,
        )
    health = app.state.policy.health()
    logger.info(
        "VLA bridge startup complete; protocol=%s policy=%s model_loaded=%s",
        PROTOCOL_VERSION,
        health.get("policy"),
        health.get("model_loaded"),
    )
    yield


app = FastAPI(title="VLA Bridge", version="2.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


def _authorized(candidate: str) -> bool:
    return hmac.compare_digest(candidate.encode("utf-8"), settings.token.encode("utf-8"))


def _websocket_authorized(websocket: WebSocket, hello: HelloMessage) -> bool:
    candidates: list[str] = []
    authorization = websocket.headers.get("authorization")
    if authorization and authorization.startswith("Bearer "):
        candidates.append(authorization[len("Bearer ") :])
    query_token = websocket.query_params.get("token")
    if query_token:
        candidates.append(query_token)
    if hello.token:
        candidates.append(hello.token)
    return any(_authorized(candidate) for candidate in candidates)


async def require_bearer(authorization: str | None = Header(default=None)) -> None:
    prefix = "Bearer "
    if authorization is None or not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="missing bearer token")
    if not _authorized(authorization[len(prefix) :]):
        raise HTTPException(status_code=403, detail="invalid bearer token")


def _error(code: str, message: str, request_id: str | None = None) -> dict[str, Any]:
    response: dict[str, Any] = {"type": "error", "code": code, "message": message}
    if request_id is not None:
        response["request_id"] = request_id
    return response


def _validation_message(exc: ValidationError) -> str:
    first = exc.errors(include_url=False)[0]
    location = ".".join(str(part) for part in first["loc"])
    return f"{location}: {first['msg']}" if location else first["msg"]


def _policy_request(message: StateMessage | HttpActionRequest) -> PolicyRequest:
    request_id = message.request_id or f"server-{uuid.uuid4()}"
    images = None
    if message.images is not None:
        images = {
            name: ImageInput(encoding=image.encoding, data=image.data)
            for name, image in message.images.items()
        }
    return PolicyRequest(
        request_id=request_id,
        instruction=message.instruction,
        state=message.state,
        images=images,
        metadata={
            "episode_id": message.episode_id,
            "step": message.step,
            "timestamp_ms": message.timestamp_ms,
            "protocol": message.protocol,
            "full_state": message.full_state,
        },
    )


async def _predict(message: StateMessage | HttpActionRequest) -> dict[str, Any]:
    act = getattr(app.state.policy, "act", None)
    if act is not None:
        async with app.state.policy_lock:
            response = await asyncio.wait_for(
                act(message.model_dump()),
                timeout=settings.inference_timeout_s,
            )
        if not isinstance(response, dict):
            raise TypeError("policy.act() must return a dict")
        return response

    predict_request = getattr(app.state.policy, "predict_request", None)
    if predict_request is not None:
        async with app.state.policy_lock:
            response = await asyncio.wait_for(
                predict_request(message.model_dump()),
                timeout=settings.inference_timeout_s,
            )
        if not isinstance(response, dict):
            raise TypeError("policy.predict_request() must return a dict")
        return response

    started = time.perf_counter()
    request = _policy_request(message)
    async with app.state.policy_lock:
        result = await asyncio.wait_for(
            app.state.policy.predict(request),
            timeout=settings.inference_timeout_s,
        )

    response: dict[str, Any] = {
        "type": "action",
        "protocol": PROTOCOL_VERSION,
        "request_id": request.request_id,
        "action": result.action,
        "raw_action_shape": result.raw_action_shape,
        "action_dtype": result.action_dtype,
        "model": result.model_name,
        "inference_ms": result.inference_ms,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "input_mode": result.input_mode,
        "timestamp_ms": int(time.time() * 1000),
    }
    if result.peak_cuda_memory_mb is not None:
        response["peak_cuda_memory_mb"] = result.peak_cuda_memory_mb
    if settings.include_action_chunk:
        response["action_chunk"] = result.action_chunk
    return response


async def _receive_json(websocket: WebSocket, timeout_s: float | None = None) -> Any:
    receive = websocket.receive_text()
    raw = await asyncio.wait_for(receive, timeout=timeout_s) if timeout_s else await receive
    if len(raw.encode("utf-8")) > settings.max_message_bytes:
        raise ValueError("MESSAGE_TOO_LARGE")
    return json.loads(raw)


@app.get("/healthz")
async def healthz(response: Response) -> dict[str, Any]:
    policy_health = app.state.policy.health()
    healthy = bool(policy_health.get("model_loaded")) and bool(app.state.policy.ready)
    response.status_code = 200 if healthy else 503
    return {
        "status": "ok" if healthy else "error",
        "protocol": PROTOCOL_VERSION,
        **policy_health,
    }


@app.post("/v1/action", dependencies=[Depends(require_bearer)], response_model=None)
async def http_action(message: HttpActionRequest) -> dict[str, Any] | JSONResponse:
    try:
        return await _predict(message)
    except PolicyInputError as exc:
        return JSONResponse(status_code=400, content=_error("INVALID_INPUT", str(exc), message.request_id))
    except PolicyNotReadyError as exc:
        return JSONResponse(status_code=503, content=_error("MODEL_NOT_READY", str(exc), message.request_id))
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content=_error("INFERENCE_TIMEOUT", "inference timed out", message.request_id),
        )
    except Exception:
        logger.exception("HTTP inference failed for request_id=%s", message.request_id)
        return JSONResponse(
            status_code=500,
            content=_error("INFERENCE_ERROR", "inference failed", message.request_id),
        )


@app.websocket("/v1/ws")
async def websocket_action(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        try:
            payload = await _receive_json(websocket, settings.hello_timeout_s)
            hello = HelloMessage.model_validate(payload)
        except asyncio.TimeoutError:
            await websocket.send_json(_error("HELLO_TIMEOUT", "hello was not received in time"))
            await websocket.close(code=4408)
            return
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            message = _validation_message(exc) if isinstance(exc, ValidationError) else str(exc)
            await websocket.send_json(_error("BAD_HELLO", message))
            await websocket.close(code=4400)
            return

        if not _websocket_authorized(websocket, hello):
            await websocket.send_json(_error("UNAUTHORIZED", "invalid token"))
            await websocket.close(code=4401)
            return

        policy_health = app.state.policy.health()
        client = (hello.client or "unknown").replace("\n", " ").replace("\r", " ")
        logger.info("[WS] connected client=%s", client)
        await websocket.send_json(
            {
                "type": "ready",
                "protocol": policy_health.get("protocol", PROTOCOL_VERSION),
                "action_dim": app.state.policy.output_action_dim,
                "policy": policy_health.get("policy"),
                "model": policy_health.get("model"),
                "model_loaded": policy_health.get("model_loaded", False),
                "action_space": policy_health.get("action_space"),
            }
        )

        while True:
            try:
                payload = await _receive_json(websocket)
            except json.JSONDecodeError:
                await websocket.send_json(_error("INVALID_JSON", "message is not valid JSON"))
                continue
            except ValueError as exc:
                await websocket.send_json(_error(str(exc), "message exceeds the configured limit"))
                await websocket.close(code=1009)
                return

            message_type = payload.get("type") if isinstance(payload, dict) else None
            try:
                if message_type == "state":
                    message = StateMessage.model_validate(payload)
                    await websocket.send_json(await _predict(message))
                elif message_type == "reset":
                    message = ResetMessage.model_validate(payload)
                    async with app.state.policy_lock:
                        await app.state.policy.reset(message.episode_id)
                    await websocket.send_json(
                        {
                            "type": "reset_ack",
                            "request_id": message.request_id,
                            "episode_id": message.episode_id,
                        }
                    )
                elif message_type == "ping":
                    message = PingMessage.model_validate(payload)
                    await websocket.send_json(
                        {
                            "type": "pong",
                            "request_id": message.request_id,
                            "timestamp_ms": int(time.time() * 1000),
                        }
                    )
                else:
                    await websocket.send_json(_error("UNKNOWN_TYPE", "expected state, reset, or ping"))
            except ValidationError as exc:
                request_id = payload.get("request_id") if isinstance(payload, dict) else None
                await websocket.send_json(
                    _error("INVALID_MESSAGE", _validation_message(exc), request_id)
                )
            except PolicyInputError as exc:
                request_id = payload.get("request_id") if isinstance(payload, dict) else None
                await websocket.send_json(_error("INVALID_INPUT", str(exc), request_id))
            except PolicyNotReadyError as exc:
                request_id = payload.get("request_id") if isinstance(payload, dict) else None
                await websocket.send_json(_error("MODEL_NOT_READY", str(exc), request_id))
            except asyncio.TimeoutError:
                request_id = payload.get("request_id") if isinstance(payload, dict) else None
                await websocket.send_json(
                    _error("INFERENCE_TIMEOUT", "inference timed out", request_id)
                )
            except Exception:
                request_id = payload.get("request_id") if isinstance(payload, dict) else None
                logger.exception("WebSocket request failed for request_id=%s", request_id)
                await websocket.send_json(
                    _error("INFERENCE_ERROR", "inference failed", request_id)
                )
    except WebSocketDisconnect:
        return
