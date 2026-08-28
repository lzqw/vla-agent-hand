"""Runtime settings for the Rabo VLA client."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

DEFAULT_PUBLIC_BASE = "https://listprice-demonstrates-saying-magazines.trycloudflare.com"
HTTP_URL = os.getenv("VLA_HTTP_URL", f"{DEFAULT_PUBLIC_BASE}/v1/action")
WS_URL = os.getenv(
    "VLA_WS_URL",
    "wss://listprice-demonstrates-saying-magazines.trycloudflare.com/v1/ws",
)
HEALTH_URL = os.getenv("VLA_HEALTH_URL", f"{DEFAULT_PUBLIC_BASE}/healthz")

TRANSPORT = os.getenv("VLA_TRANSPORT", "auto").strip().lower()
WS_AUTH_MODE = os.getenv("VLA_WS_AUTH_MODE", "auto").strip().lower()
TOKEN_ENV = os.getenv("VLA_TOKEN", "").strip()
TOKEN_FILE = Path(
    os.getenv("VLA_TOKEN_FILE", str(PROJECT_ROOT / ".vla_token"))
).expanduser()

MODE = os.getenv("RABO_MODE", "sim")
SENSOR_BACKEND = os.getenv("RABO_SENSOR_BACKEND", "ros2")
CONTROL_HZ = float(os.getenv("VLA_CONTROL_HZ", "5.0"))
READY_TIMEOUT_S = float(os.getenv("VLA_READY_TIMEOUT_S", "30"))
NETWORK_TIMEOUT_S = float(os.getenv("VLA_NETWORK_TIMEOUT_S", "5"))
STARTUP_DELAY_S = float(os.getenv("VLA_STARTUP_DELAY_S", "1"))
MAX_CYCLES = int(os.getenv("VLA_MAX_CYCLES", "0"))
JPEG_QUALITY = int(os.getenv("VLA_JPEG_QUALITY", "80"))

EXECUTE_ACTIONS = os.getenv("VLA_EXECUTE_ACTIONS", "1") == "1"
EXECUTE_HAND_ACTIONS = os.getenv("VLA_EXECUTE_HAND_ACTIONS", "1") == "1"
ACTION_SPACE = os.getenv("VLA_ACTION_SPACE", "arm_joint_position_14d")
PROTOCOL = os.getenv("VLA_PROTOCOL", "rabo_command_v1")
MAX_ARM_STEP_RAD = float(os.getenv("VLA_MAX_ARM_STEP_RAD", "0.04"))

CAMERA_NAMES = tuple(
    part.strip()
    for part in os.getenv(
        "VLA_CAMERA_NAMES", "cam_high,cam_left_wrist,cam_right_wrist"
    ).split(",")
    if part.strip()
)
INSTRUCTION = os.getenv("VLA_INSTRUCTION", "").strip()

LOCAL_PREPOSITION = os.getenv("VLA_LOCAL_PREPOSITION", "1") == "1"
RESET_FIXED_SCENE = os.getenv("VLA_RESET_FIXED_SCENE", "1") == "1"

if TRANSPORT not in {"auto", "ws", "http"}:
    raise ValueError("VLA_TRANSPORT must be auto/ws/http")
if WS_AUTH_MODE not in {"auto", "bearer", "query", "hello", "none"}:
    raise ValueError("VLA_WS_AUTH_MODE must be auto/bearer/query/hello/none")
if CONTROL_HZ <= 0:
    raise ValueError("VLA_CONTROL_HZ must be positive")
if not 1 <= JPEG_QUALITY <= 100:
    raise ValueError("VLA_JPEG_QUALITY must be in [1,100]")
if ACTION_SPACE != "arm_joint_position_14d":
    raise ValueError("VLA_ACTION_SPACE must be arm_joint_position_14d")
