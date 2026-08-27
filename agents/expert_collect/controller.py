"""Collect fixed-scene expert command episodes for the remote 4080 policy."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2

from drivers import RaboDevices, RemoteCommandExecutor, Ros2SensorBackend, load_config
from .expert_program import FIXED_NUT_POSES, build_expert_program

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_TEXT = "双臂协作依次抓取B、C、A螺母，由右手递交左手并放入目标区"


def _next_episode_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    used = []
    for p in root.glob("episode_*"):
        try:
            used.append(int(p.name.split("_")[-1]))
        except ValueError:
            pass
    target = root / f"episode_{max(used, default=-1) + 1:06d}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_snapshot(episode_dir: Path, step: int, snapshot) -> dict[str, Any]:
    image_paths: dict[str, str] = {}
    for name, image in snapshot.images.items():
        rel = Path("images") / name / f"{step:04d}.jpg"
        target = episode_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise RuntimeError(f"failed to save image: {target}")
        image_paths[name] = rel.as_posix()
    return {
        "state": [float(v) for v in snapshot.state],
        "full_state": [float(v) for v in snapshot.full_state],
        "images": image_paths,
        "image_stamps": {k: float(v) for k, v in snapshot.image_stamps.items()},
        "camera_reused": {k: bool(v) for k, v in snapshot.camera_reused.items()},
        "monotonic_time": float(snapshot.monotonic_time),
    }


def collect_one() -> Path:
    config = load_config(PROJECT_ROOT / "config.yaml")
    devices = RaboDevices(config, mode=str(config.rabo.get("mode", "sim")))
    sensors = Ros2SensorBackend(config, joint_reader=devices.read_full_state)
    executor = RemoteCommandExecutor(devices, trace=True)
    output_root = Path(os.getenv("EXPERT_DATA_ROOT", str(PROJECT_ROOT / "expert_data"))).expanduser()
    episode_dir = _next_episode_dir(output_root)
    stop_event = threading.Event()
    program = build_expert_program()
    rows: list[dict[str, Any]] = []
    started = time.time()

    try:
        print(f"[expert] output={episode_dir}", flush=True)
        sensors.start()
        sensors.wait_ready(float(os.getenv("EXPERT_READY_TIMEOUT_S", "30")))
        devices.reset_fixed_scene()
        devices.pre_position()
        sensors.reset_clock()

        previous_command: dict[str, Any] | None = None
        for step, item in enumerate(program):
            phase = str(item["phase"])
            command = item["command"]
            snapshot = sensors.sample_next(0.0, 0.0, stop_event)
            if snapshot is None:
                raise RuntimeError(f"failed to capture observation before step {step}")
            record: dict[str, Any] = {
                "step": step,
                "phase": phase,
                "instruction": TASK_TEXT,
                "observation": _save_snapshot(episode_dir, step, snapshot),
                "previous_command": previous_command,
                "command": command,
                "execution_success": None,
                "execution_elapsed_s": None,
                "execution_error": "",
            }
            t0 = time.monotonic()
            try:
                result = executor.execute(command)
            except BaseException as exc:
                record["execution_success"] = False
                record["execution_elapsed_s"] = time.monotonic() - t0
                record["execution_error"] = repr(exc)
                rows.append(record)
                raise
            record["execution_success"] = True
            record["execution_elapsed_s"] = time.monotonic() - t0
            rows.append(record)
            previous_command = command
            print(f"[expert] step={step:02d} phase={phase} action={command['action_type']}", flush=True)
            if result.get("done"):
                break

        (episode_dir / "steps.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        _write_json(episode_dir / "expert_program.json", {
            "format": "rabo_expert_program_v1",
            "protocol": "rabo_command_v1",
            "task": TASK_TEXT,
            "fixed_nut_poses": {k: list(v) for k, v in FIXED_NUT_POSES.items()},
            "num_steps": len(rows),
            "commands": [
                {"step": row["step"], "phase": row["phase"], "command": row["command"]}
                for row in rows
            ],
        })
        _write_json(episode_dir / "meta.json", {
            "format": "rabo_expert_command_dataset_v1",
            "protocol": "rabo_command_v1",
            "task": TASK_TEXT,
            "num_steps": len(rows),
            "sdk_sequence_completed": bool(rows and rows[-1]["command"].get("action_type") == "done"),
            "physical_success_requires_visual_confirmation": True,
            "elapsed_s": time.time() - started,
            "primary_camera": config.primary_camera,
            "camera_names": list(config.camera_topics),
            "state_dim": len(config.state_names),
            "full_state_dim": len(config.full_state_names),
            "fixed_nut_poses": {k: list(v) for k, v in FIXED_NUT_POSES.items()},
        })
        print(f"[expert] saved {len(rows)} command steps", flush=True)
        print(f"[expert] 4080-ready: {episode_dir / 'expert_program.json'}", flush=True)
        return episode_dir
    except BaseException as exc:
        if rows:
            (episode_dir / "steps.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
        _write_json(episode_dir / "meta.json", {
            "format": "rabo_expert_command_dataset_v1",
            "protocol": "rabo_command_v1",
            "task": TASK_TEXT,
            "num_steps": len(rows),
            "sdk_sequence_completed": False,
            "failure_reason": repr(exc),
            "physical_success_requires_visual_confirmation": True,
        })
        raise
    finally:
        sensors.close()
        devices.shutdown()


def run() -> None:
    episodes = max(1, int(os.getenv("EXPERT_EPISODES", "1")))
    for index in range(episodes):
        print(f"[expert] collecting episode {index + 1}/{episodes}", flush=True)
        collect_one()
