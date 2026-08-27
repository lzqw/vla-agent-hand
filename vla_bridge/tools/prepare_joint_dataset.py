"""Prepare pure-joint reference/BC artifacts from one LeRobot episode bundle.

Expected bundle layout (see the bundle exported from episode_000001):

  <episode_root>/data/episode_000001.parquet
  <episode_root>/videos/cam_high/episode_000001.mp4
  <episode_root>/videos/cam_left_wrist/episode_000001.mp4
  <episode_root>/videos/cam_right_wrist/episode_000001.mp4

Outputs:
  reference_v1.npz  -- full_state[t] -> full_state[t+1] reference trajectory
  bc_episode_v1.npz -- 3 RGB + proprio -> next full_state regression cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_parquet(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required: pip install pyarrow") from exc

    table = pq.read_table(
        path,
        columns=["observation.state", "observation.full_state", "phase"],
    )
    state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    full = np.asarray(table.column("observation.full_state").to_pylist(), dtype=np.float32)
    phases = [str(v) for v in table.column("phase").to_pylist()]
    if state.ndim != 2 or state.shape[1] != 26:
        raise ValueError(f"observation.state must be [N,26], got {state.shape}")
    if full.ndim != 2 or full.shape[1] != 36 or len(full) != len(state):
        raise ValueError(f"observation.full_state must be [N,36], got {full.shape}")
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
            # Store RGB uint8 so training/inference preprocessing is unambiguous.
            result.append(frame[:, :, ::-1].copy())
    finally:
        capture.release()
    if len(result) != frames:
        raise ValueError(f"video {path} has {len(result)} decoded frames, expected {frames}")
    return np.stack(result, axis=0).astype(np.uint8, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_root", type=Path)
    parser.add_argument("--episode-index", default="000001")
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
        "cam_high": root / "videos" / "cam_high" / f"episode_{index}.mp4",
        "cam_left_wrist": root / "videos" / "cam_left_wrist" / f"episode_{index}.mp4",
        "cam_right_wrist": root / "videos" / "cam_right_wrist" / f"episode_{index}.mp4",
    }
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    for path in videos.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    state, full, phases = _read_parquet(parquet)
    n = len(full)
    target_full = np.concatenate([full[1:], full[-1:]], axis=0).astype(np.float32)
    progress = np.linspace(0.0, 1.0, n, dtype=np.float32)

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference_path = output / "reference_v1.npz"
    np.savez_compressed(
        reference_path,
        reference_state=state,
        reference_full_state=full,
        target_full_state=target_full,
        progress=progress,
    )

    images = {
        name: _read_video(path, n, args.image_width, args.image_height)
        for name, path in videos.items()
    }
    bc_path = output / "bc_episode_v1.npz"
    np.savez_compressed(
        bc_path,
        state=state,
        full_state=full,
        target_full_state=target_full,
        progress=progress,
        cam_high=images["cam_high"],
        cam_left_wrist=images["cam_left_wrist"],
        cam_right_wrist=images["cam_right_wrist"],
    )

    meta = {
        "format": "rabo_joint_episode_v1",
        "episode_index": index,
        "frames": n,
        "input_fps": 5,
        "state_dim": 26,
        "full_state_dim": 36,
        "action_dim": 36,
        "action_definition": "target_full_state[t] = observation.full_state[t+1]",
        "cameras": list(videos),
        "bc_image_size": [args.image_height, args.image_width],
        "reference_file": str(reference_path),
        "bc_cache_file": str(bc_path),
        "phases": sorted(set(phases)),
    }
    (output / "dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
