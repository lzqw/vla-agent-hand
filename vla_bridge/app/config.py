from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path


PROTOCOL_VERSION = "vla-bridge.v1"


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _choice(name: str, default: str, choices: set[str]) -> str:
    value = os.getenv(name, default).strip().lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}, got {value!r}")
    return value


def _load_or_create_token() -> str:
    configured = os.getenv("VLA_BRIDGE_TOKEN")
    if configured:
        return configured

    token_path = Path(
        os.getenv(
            "VLA_BRIDGE_TOKEN_FILE",
            str(Path.home() / ".config" / "vla-bridge" / "token"),
        )
    ).expanduser()
    token_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)

    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        token = secrets.token_urlsafe(32)
        try:
            fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            token = token_path.read_text(encoding="utf-8").strip()
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(token + "\n")

    if len(token) < 24:
        raise ValueError("VLA bridge token must contain at least 24 characters")
    token_path.chmod(0o600)
    return token


@dataclass(frozen=True)
class Settings:
    token: str
    action_dim: int
    hello_timeout_s: float
    inference_timeout_s: float
    max_message_bytes: int
    policy: str
    expert_program_path: Path
    joint_reference_path: Path
    joint_initial_search: int
    joint_forward_window: int
    bc_joint_model_dir: Path
    model_backend: str
    model_device: str
    model_dtype: str
    model_path: Path
    action_expert_path: Path
    dexvla_repo_path: Path
    dexvla_dummy_test: bool
    dexvla_dummy_cameras: int
    output_mode: str
    include_action_chunk: bool


def load_settings() -> Settings:
    runtime = Path.home() / "vla_runtime"
    bridge = Path.home() / "vla_bridge"
    return Settings(
        token=_load_or_create_token(),
        action_dim=_positive_int("VLA_BRIDGE_ACTION_DIM", 7),
        hello_timeout_s=_positive_float("VLA_BRIDGE_HELLO_TIMEOUT_S", 10.0),
        inference_timeout_s=_positive_float("VLA_BRIDGE_INFERENCE_TIMEOUT_S", 300.0),
        max_message_bytes=_positive_int("VLA_BRIDGE_MAX_MESSAGE_BYTES", 16 * 1024 * 1024),
        policy=_choice(
            "VLA_POLICY",
            "zero",
            {
                "zero",
                "dexvla",
                "expert_lookup",
                "rabo_vla",
                "joint_vla",
                "bc_joint_vla",
            },
        ),
        expert_program_path=Path(
            os.getenv(
                "EXPERT_PROGRAM_PATH",
                str(bridge / "data" / "expert_program.json"),
            )
        ).expanduser(),
        joint_reference_path=Path(
            os.getenv(
                "JOINT_REFERENCE_PATH",
                str(bridge / "data" / "joint" / "reference_v1.npz"),
            )
        ).expanduser(),
        joint_initial_search=_positive_int("JOINT_INITIAL_SEARCH", 250),
        joint_forward_window=_positive_int("JOINT_FORWARD_WINDOW", 80),
        bc_joint_model_dir=Path(
            os.getenv(
                "BC_JOINT_MODEL_DIR",
                str(bridge / "models" / "rabo_bc_joint_v1"),
            )
        ).expanduser(),
        model_backend=_choice("MODEL_BACKEND", "dexvla", {"dexvla"}),
        model_device=_choice("MODEL_DEVICE", "cuda", {"cuda"}),
        model_dtype=_choice(
            "MODEL_DTYPE", "bfloat16", {"bfloat16", "bf16", "float16", "fp16"}
        ),
        model_path=Path(
            os.getenv("MODEL_PATH", str(runtime / "weights" / "qwen2_vl_2b"))
        ).expanduser(),
        action_expert_path=Path(
            os.getenv(
                "ACTION_EXPERT_PATH",
                str(runtime / "weights" / "scaledp_l" / "open_scale_dp_l_backbone.ckpt"),
            )
        ).expanduser(),
        dexvla_repo_path=Path(
            os.getenv("DEXVLA_REPO_PATH", str(runtime / "third_party" / "DexVLA"))
        ).expanduser(),
        dexvla_dummy_test=_boolean("DEXVLA_DUMMY_TEST", False),
        dexvla_dummy_cameras=_bounded_int("DEXVLA_DUMMY_CAMERAS", 1, 1, 3),
        output_mode=_choice("VLA_OUTPUT_MODE", "raw", {"raw", "compat7"}),
        include_action_chunk=_boolean("VLA_INCLUDE_ACTION_CHUNK", True),
    )
