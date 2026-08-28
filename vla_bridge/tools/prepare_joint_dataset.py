"""Prepare dense 14D arm BC data plus command-Expert hand events.

Inputs:
  * one 5 Hz LeRobot episode (Parquet + three MP4 files), and
  * one successful command-level Expert steps.jsonl.

The dense episode supplies arm targets. The command episode supplies only the
validated O6 hand/wait semantics. Request step is never stored as a model input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ARM_DIM = 14
FULL_STATE_DIM = 36
STATE_DIM = 26
HAND_ACTION_TYPES = {
    "right_hand_clench",
    "left_hand_clench",
    "right_hand_grasp_force",
    "left_hand_grasp_force",
    "right_hand_release",
    "left_hand_release",
    "wait",
}


def _read_parquet(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required: pip install pyarrow") from exc

    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    columns = ["observation.state", "observation.full_state"]
    if "phase" in schema_names:
        columns.append("phase")
    table = pq.read_table(path, columns=columns)
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    full = np.asarray(table.column("observation.full_state").to_pylist(), dtype=np.float32)
    phases = (
        [str(value) for value in table.column("phase").to_pylist()]
        if "phase" in table.column_names
        else [""] * len(full)
    )
    if state.ndim != 2 or state.shape[1] != STATE_DIM:
        raise ValueError(f"observation.state must be [N,{STATE_DIM}], got {state.shape}")
    if full.ndim != 2 or full.shape != (len(state), FULL_STATE_DIM):
        raise ValueError(
            f"observation.full_state must be [N,{FULL_STATE_DIM}], got {full.shape}"
        )
    if len(full) < 2:
        raise ValueError("dense episode must contain at least two frames")
    if not np.isfinite(state).all() or not np.isfinite(full).all():
        raise ValueError("parquet contains non-finite joint values")
    return state, full, phases


def _read_video(path: Path, frames: int, width: int, height: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open video: {path}")
    result: list[np.ndarray] = []
    try:
        while len(result) < frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            result.append(frame[:, :, ::-1].copy())
    finally:
        capture.release()
    if len(result) != frames:
        raise ValueError(f"video {path} has {len(result)} decoded frames, expected {frames}")
    return np.stack(result, axis=0).astype(np.uint8, copy=False)


def _read_expert_steps(path: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("command Expert steps.jsonl is empty")
    arms: list[np.ndarray] = []
    for index, record in enumerate(records):
        if record.get("execution_success") is not True:
            raise ValueError(f"command Expert step {index} is not execution_success=true")
        observation = record.get("observation")
        command = record.get("command")
        if not isinstance(observation, dict) or not isinstance(command, dict):
            raise ValueError(f"command Expert step {index} is missing observation/command")
        full = np.asarray(observation.get("full_state"), dtype=np.float32)
        if full.shape != (FULL_STATE_DIM,) or not np.isfinite(full).all():
            raise ValueError(
                f"command Expert step {index} full_state must contain {FULL_STATE_DIM} values"
            )
        if not isinstance(command.get("action_type"), str):
            raise ValueError(f"command Expert step {index} command has no action_type")
        arms.append(full[:ARM_DIM])
    return records, np.stack(arms).astype(np.float32, copy=False)


def _monotonic_dtw_alignment(query: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map each sparse Expert state to a monotonic dense reference index."""

    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("alignment inputs must be [M,D] and [N,D]")
    m, n = len(query), len(reference)
    scale = np.maximum(np.std(reference, axis=0), np.float32(0.03))
    normalized = (query[:, None, :] - reference[None, :, :]) / scale[None, None, :]
    local_cost = np.mean(normalized * normalized, axis=2).astype(np.float64)

    cumulative = np.full((m, n), np.inf, dtype=np.float64)
    predecessor = np.zeros((m, n), dtype=np.uint8)
    cumulative[0, 0] = local_cost[0, 0]
    for j in range(1, n):
        cumulative[0, j] = cumulative[0, j - 1] + local_cost[0, j]
        predecessor[0, j] = 2  # left
    for i in range(1, m):
        cumulative[i, 0] = cumulative[i - 1, 0] + local_cost[i, 0]
        predecessor[i, 0] = 1  # up
    for i in range(1, m):
        for j in range(1, n):
            choices = (
                cumulative[i - 1, j - 1],
                cumulative[i - 1, j],
                cumulative[i, j - 1],
            )
            move = int(np.argmin(choices))
            cumulative[i, j] = choices[move] + local_cost[i, j]
            predecessor[i, j] = move  # diagonal=0, up=1, left=2

    path: list[tuple[int, int]] = []
    i, j = m - 1, n - 1
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        move = int(predecessor[i, j])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()

    candidates: list[list[int]] = [[] for _ in range(m)]
    for query_index, reference_index in path:
        candidates[query_index].append(reference_index)
    mapping = np.empty(m, dtype=np.int64)
    for query_index, indices in enumerate(candidates):
        mapping[query_index] = min(
            indices, key=lambda reference_index: local_cost[query_index, reference_index]
        )
    if np.any(mapping[1:] < mapping[:-1]):
        raise RuntimeError("internal error: DTW mapping is not monotonic")

    raw_delta = query - reference[mapping]
    rms_rad = np.sqrt(np.mean(raw_delta * raw_delta, axis=1)).astype(np.float32)
    return mapping, rms_rad


def _hand_events(
    records: list[dict[str, Any]], mapping: np.ndarray, rms_rad: np.ndarray
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for expert_step, record in enumerate(records):
        command = record["command"]
        if command.get("action_type") not in HAND_ACTION_TYPES:
            continue
        events.append(
            {
                "event_id": len(events),
                "expert_step": expert_step,
                "phase": record.get("phase"),
                "frame_index": int(mapping[expert_step]),
                "alignment_rms_rad": round(float(rms_rad[expert_step]), 7),
                "command": command,
            }
        )
    if not events:
        raise ValueError("command Expert contains no supported hand/wait events")
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_root", type=Path)
    parser.add_argument("--episode-index", default="000001")
    parser.add_argument(
        "--expert-steps",
        type=Path,
        default=(
            Path.home()
            / "vla_bridge"
            / "data"
            / "bc"
            / "expert_data"
            / "episode_000000"
            / "steps.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "vla_bridge" / "data" / "joint",
    )
    parser.add_argument("--image-width", type=int, default=128)
    parser.add_argument("--image-height", type=int, default=128)
    args = parser.parse_args()

    root = args.episode_root.expanduser().resolve()
    index = str(args.episode_index)
    parquet = root / "data" / f"episode_{index}.parquet"
    videos = {
        name: root / "videos" / name / f"episode_{index}.mp4" for name in CAMERA_KEYS
    }
    expert_steps = args.expert_steps.expanduser().resolve()
    for path in (parquet, expert_steps, *videos.values()):
        if not path.is_file():
            raise FileNotFoundError(path)

    state, full, phases = _read_parquet(parquet)
    records, expert_arms = _read_expert_steps(expert_steps)
    reference_arms = full[:, :ARM_DIM].astype(np.float32, copy=True)
    target_arms = np.concatenate([reference_arms[1:], reference_arms[-1:]], axis=0)
    progress = np.linspace(0.0, 1.0, len(full), dtype=np.float32)
    mapping, rms_rad = _monotonic_dtw_alignment(expert_arms, reference_arms)
    events = _hand_events(records, mapping, rms_rad)

    images = {
        name: _read_video(path, len(full), args.image_width, args.image_height)
        for name, path in videos.items()
    }

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_path = output / "arm_hand_reference_v1.npz"
    np.savez_compressed(
        reference_path,
        reference_state=state,
        reference_full_state=full,
        reference_arm_state=reference_arms,
        target_arm_state=target_arms,
        progress=progress,
    )

    bc_path = output / "bc_episode_v1.npz"
    np.savez_compressed(
        bc_path,
        state=state,
        full_state=full,
        target_arm_state=target_arms,
        progress=progress,
        cam_high=images["cam_high"],
        cam_left_wrist=images["cam_left_wrist"],
        cam_right_wrist=images["cam_right_wrist"],
    )

    hand_events_path = output / "hand_events_v1.json"
    hand_doc = {
        "format": "rabo_hand_events_v1",
        "protocol": "rabo_command_v1",
        "reference_frames": len(full),
        "alignment": "monotonic_dtw_arm_state_14d",
        "expert_records": len(records),
        "event_count": len(events),
        "source_expert_steps": str(expert_steps),
        "events": events,
    }
    hand_events_path.write_text(
        json.dumps(hand_doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    meta = {
        "format": "rabo_arm_hand_episode_v1",
        "episode_index": index,
        "frames": len(full),
        "input_fps": 5,
        "state_dim": STATE_DIM,
        "full_state_dim": FULL_STATE_DIM,
        "action_space": "arm_joint_position_14d",
        "action_dim": ARM_DIM,
        "action_order": "left_arm(7) + right_arm(7)",
        "action_definition": "target_arm_state[t] = observation.full_state[t+1,:14]",
        "last_target_definition": "target_arm_state[-1] = observation.full_state[-1,:14]",
        "controls_o6_hand_joints": False,
        "cameras": list(CAMERA_KEYS),
        "bc_image_size": [args.image_height, args.image_width],
        "reference_file": str(reference_path),
        "bc_cache_file": str(bc_path),
        "hand_events_file": str(hand_events_path),
        "hand_event_count": len(events),
        "expert_records": len(records),
        "alignment_mean_rms_rad": float(np.mean(rms_rad)),
        "alignment_max_rms_rad": float(np.max(rms_rad)),
        "phases": sorted({phase for phase in phases if phase}),
    }
    (output / "dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
