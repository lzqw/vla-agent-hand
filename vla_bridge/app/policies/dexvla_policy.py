from __future__ import annotations

import asyncio
import base64
import binascii
import io
import logging
import math
import sys
import threading
from collections.abc import Sequence
from typing import Any

from ..action_adapter import RobotActionAdapter
from ..config import Settings
from .base import BasePolicy, ImageInput, PolicyInputError, PolicyRequest, PolicyResult


logger = logging.getLogger("vla_bridge.dexvla")


class DexVLAPolicy(BasePolicy):
    name = "dexvla"

    def __init__(self, settings: Settings) -> None:
        if settings.model_backend != "dexvla":
            raise ValueError(
                f"VLA_POLICY=dexvla requires MODEL_BACKEND=dexvla, got {settings.model_backend}"
            )

        runtime_path = settings.model_path.parent.parent
        if str(runtime_path) not in sys.path:
            sys.path.insert(0, str(runtime_path))

        # Kept lazy so VLA_POLICY=zero still runs in the small bridge environment.
        from dexvla_runner import DexVLARunner, make_dummy_observation

        self.settings = settings
        self._make_dummy_observation = make_dummy_observation
        self._gpu_lock = threading.Lock()
        self._adapter = RobotActionAdapter(settings.output_mode)
        self.ready = False
        self.last_warmup: dict[str, Any] | None = None
        self.runner = DexVLARunner(
            model_path=settings.model_path,
            action_expert_path=settings.action_expert_path,
            repo_path=settings.dexvla_repo_path,
            device=settings.model_device,
            dtype=settings.model_dtype,
        )
        native_dim = int(self.runner.model.policy_head.action_dim)
        self.output_action_dim = self._adapter.output_dim(native_dim)

    async def warmup(self) -> None:
        prediction = await asyncio.to_thread(self._warmup_sync)
        self.last_warmup = {
            "raw_action_shape": prediction.raw_action_shape,
            "inference_ms": prediction.inference_ms,
            "peak_cuda_memory_mb": prediction.peak_cuda_memory_mb,
        }
        self.ready = True
        logger.info("DexVLA warmup complete: %s", self.last_warmup)

    def _warmup_sync(self):
        images, state, instruction = self._make_dummy_observation(
            camera_count=self.settings.dexvla_dummy_cameras,
            height=240,
            width=320,
        )
        with self._gpu_lock:
            return self.runner.predict(images, state, instruction)

    async def predict(self, request: PolicyRequest) -> PolicyResult:
        return await asyncio.to_thread(self._predict_sync, request)

    def _predict_sync(self, request: PolicyRequest) -> PolicyResult:
        missing = []
        if not request.images:
            missing.append("images")
        if request.state is None:
            missing.append("state")
        if not request.instruction or not request.instruction.strip():
            missing.append("instruction")

        if missing:
            if not self.settings.dexvla_dummy_test:
                raise PolicyInputError(
                    "DexVLA requires real images, state, and instruction; missing "
                    + ", ".join(missing)
                )
            images, state, instruction = self._make_dummy_observation(
                camera_count=self.settings.dexvla_dummy_cameras,
                height=240,
                width=320,
            )
            input_mode = "dummy_test"
        else:
            images = self._decode_camera_dict(request.images or {})
            state = self._coerce_state(request.state)
            instruction = request.instruction.strip()
            input_mode = "request"

        try:
            with self._gpu_lock:
                prediction = self.runner.predict(images, state, instruction)
        except (TypeError, ValueError) as exc:
            raise PolicyInputError(str(exc)) from exc

        return PolicyResult(
            action=self._adapter.adapt(prediction.action_chunk),
            action_chunk=prediction.action_chunk,
            raw_action_shape=prediction.raw_action_shape,
            inference_ms=prediction.inference_ms,
            model_name=prediction.model_name,
            action_dtype=prediction.action_dtype,
            input_mode=input_mode,
            peak_cuda_memory_mb=prediction.peak_cuda_memory_mb,
        )

    @staticmethod
    def _coerce_state(value: Any) -> list[float]:
        if isinstance(value, (str, bytes, dict)) or not isinstance(value, Sequence):
            raise PolicyInputError(
                "DexVLA Stage-1 currently requires state to be a flat JSON numeric array"
            )
        try:
            state = [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise PolicyInputError("state must contain only numeric values") from exc
        if any(not math.isfinite(item) for item in state):
            raise PolicyInputError("state contains NaN or Inf")
        return state

    @classmethod
    def _decode_camera_dict(cls, cameras: dict[str, ImageInput]) -> list[Any]:
        if not cameras:
            raise PolicyInputError("images must contain at least one camera")
        if len(cameras) > 3:
            raise PolicyInputError(f"DexVLA accepts at most 3 cameras, got {len(cameras)}")

        priority = {
            "front": 0,
            "top": 0,
            "wrist_left": 1,
            "left_wrist": 1,
            "wrist_right": 2,
            "right_wrist": 2,
        }
        ordered = sorted(
            enumerate(cameras.items()),
            key=lambda item: (priority.get(item[1][0], 3), item[0]),
        )
        return [cls._decode_jpeg(name, payload) for _, (name, payload) in ordered]

    @staticmethod
    def _decode_jpeg(camera_name: str, payload: ImageInput):
        if payload.encoding != "jpeg_base64":
            raise PolicyInputError(
                f"images.{camera_name}.encoding must be jpeg_base64"
            )
        try:
            raw = base64.b64decode(payload.data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise PolicyInputError(f"images.{camera_name}.data is not valid base64") from exc
        if not raw:
            raise PolicyInputError(f"images.{camera_name}.data is empty")

        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(io.BytesIO(raw)) as source:
                image_format = source.format
                source.load()
                if image_format not in {"JPEG", "JPG"}:
                    raise PolicyInputError(
                        f"images.{camera_name} is {image_format or 'unknown'}, expected JPEG"
                    )
                if source.width * source.height > 16_777_216:
                    raise PolicyInputError(f"images.{camera_name} exceeds 16 megapixels")
                return source.convert("RGB")
        except PolicyInputError:
            raise
        except (UnidentifiedImageError, OSError) as exc:
            raise PolicyInputError(f"images.{camera_name} is not a valid JPEG") from exc

    def health(self) -> dict[str, Any]:
        health = self.runner.health()
        health.update(
            {
                "policy": self.name,
                "model_loaded": self.ready,
                "output_mode": self.settings.output_mode,
                "output_action_dim": self.output_action_dim,
                "dummy_test": self.settings.dexvla_dummy_test,
                "warmup": self.last_warmup,
            }
        )
        return health
