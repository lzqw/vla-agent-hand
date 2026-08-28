"""Shared dense-arm alignment and one-shot hand-event scheduling.

The dense reference contains only the two A7 arms as executable numeric
targets. O6 hand actions remain the original validated SDK commands. The live
observation is aligned to the recorded arm trajectory, then a short event-aware
lookahead is used to choose the next arm target without skipping a hand event.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ACTION_SPACE = "arm_joint_position_14d"
ARM_DIM = 14
FULL_STATE_DIM = 36


@dataclass(frozen=True)
class AlignmentResult:
    reference_index: int
    action_reference_index: int
    match_distance: float
    target_arm: np.ndarray
    hand_command: dict[str, Any] | None
    hand_event_id: int | None
    done: bool


class ArmHandReference:
    """Align live arm state to a dense trajectory and schedule hand commands."""

    def __init__(
        self,
        reference_path: Path,
        hand_events_path: Path,
        *,
        initial_search: int = 250,
        forward_window: int = 80,
        target_tolerance_rad: float = 0.01,
        lookahead_frames: int = 1,
    ) -> None:
        self.reference_path = reference_path.expanduser().resolve()
        self.hand_events_path = hand_events_path.expanduser().resolve()

        try:
            raw = np.load(self.reference_path, allow_pickle=False)
        except FileNotFoundError as exc:
            raise ValueError(f"arm/hand reference not found: {self.reference_path}") from exc

        required = {"reference_arm_state", "target_arm_state"}
        missing = required.difference(raw.files)
        if missing:
            raise ValueError(f"arm/hand reference missing arrays: {sorted(missing)}")
        reference = np.asarray(raw["reference_arm_state"], dtype=np.float32)
        target = np.asarray(raw["target_arm_state"], dtype=np.float32)
        if reference.ndim != 2 or reference.shape[1] != ARM_DIM:
            raise ValueError(f"reference_arm_state must be [N,{ARM_DIM}], got {reference.shape}")
        if target.shape != reference.shape:
            raise ValueError(
                f"target_arm_state must match reference shape {reference.shape}, got {target.shape}"
            )
        if len(reference) < 2:
            raise ValueError("arm/hand reference must contain at least two frames")
        if not np.isfinite(reference).all() or not np.isfinite(target).all():
            raise ValueError("arm/hand reference contains non-finite values")

        try:
            event_doc = json.loads(self.hand_events_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"hand events not found: {self.hand_events_path}") from exc
        if event_doc.get("format") != "rabo_hand_events_v1":
            raise ValueError("hand events format must be rabo_hand_events_v1")
        if int(event_doc.get("reference_frames", -1)) != len(reference):
            raise ValueError("hand events reference_frames does not match reference trajectory")

        raw_events = event_doc.get("events")
        if not isinstance(raw_events, list):
            raise ValueError("hand events events must be an array")
        events: list[dict[str, Any]] = []
        previous_frame = -1
        for expected_id, item in enumerate(raw_events):
            if not isinstance(item, dict):
                raise ValueError("each hand event must be an object")
            event_id = int(item.get("event_id", -1))
            frame_index = int(item.get("frame_index", -1))
            command = item.get("command")
            if event_id != expected_id:
                raise ValueError("hand event ids must be contiguous from zero")
            if frame_index < 0 or frame_index < previous_frame or frame_index >= len(reference):
                raise ValueError("hand event frame indices must be monotonic and in range")
            if not isinstance(command, dict) or not isinstance(command.get("action_type"), str):
                raise ValueError("hand event command must contain action_type")
            previous_frame = frame_index
            events.append(
                {
                    "event_id": event_id,
                    "frame_index": frame_index,
                    "expert_step": int(item.get("expert_step", -1)),
                    "phase": item.get("phase"),
                    "command": command,
                }
            )

        self.reference = reference
        self.target = target
        self.events = tuple(events)
        self.num_frames = int(reference.shape[0])
        self.initial_search = max(1, min(int(initial_search), self.num_frames))
        self.forward_window = max(2, int(forward_window))
        if not np.isfinite(target_tolerance_rad) or target_tolerance_rad <= 0.0:
            raise ValueError("target_tolerance_rad must be positive and finite")
        self.target_tolerance_rad = float(target_tolerance_rad)
        self.lookahead_frames = max(1, int(lookahead_frames))
        self.scale = np.maximum(np.std(reference, axis=0), np.float32(0.03)).astype(
            np.float32
        )

        self._cursor: dict[str, int] = {}
        self._fired: dict[str, set[int]] = {}
        self._lock = threading.Lock()

    def _nearest(self, episode_id: str, current_arm: np.ndarray) -> tuple[int, float]:
        previous = self._cursor.get(episode_id, -1)
        if previous < 0:
            start = 0
            end = self.initial_search
        else:
            start = max(0, previous - 2)
            end = min(self.num_frames, previous + self.forward_window + 1)

        candidates = self.reference[start:end]
        normalized = (candidates - current_arm[None, :]) / self.scale[None, :]
        distances = np.mean(normalized * normalized, axis=1)
        local = int(np.argmin(distances))
        matched = start + local
        selected = max(previous, matched)

        # Repeated or near-static frames can have identical matching cost. Once
        # the live arm reaches the one-step recorded target for the current
        # cursor, allow the cursor to advance. The request step is never used.
        if previous >= 0 and previous < self.num_frames - 1:
            target_error = float(np.max(np.abs(self.target[previous] - current_arm)))
            if target_error <= self.target_tolerance_rad:
                selected = max(selected, previous + 1)
        selected = min(selected, self.num_frames - 1)

        selected_delta = (self.reference[selected] - current_arm) / self.scale
        distance = float(np.mean(selected_delta * selected_delta))
        self._cursor[episode_id] = selected
        return selected, distance

    def _next_due_event(
        self, episode_id: str, reference_index: int
    ) -> tuple[dict[str, Any] | None, int | None]:
        fired = self._fired.setdefault(episode_id, set())
        for event in self.events:
            event_id = int(event["event_id"])
            if event_id not in fired and int(event["frame_index"]) <= reference_index:
                fired.add(event_id)
                return dict(event["command"]), event_id
        return None, None

    def _next_pending_event_frame(self, episode_id: str) -> int | None:
        fired = self._fired.setdefault(episode_id, set())
        for event in self.events:
            if int(event["event_id"]) not in fired:
                return int(event["frame_index"])
        return None

    def _action_reference_index(
        self,
        episode_id: str,
        reference_index: int,
        hand_command: dict[str, Any] | None,
    ) -> int:
        # When a hand command is emitted, hold the arms at the aligned reference
        # frame for that cycle. This prevents a lookahead arm command from moving
        # through a grasp/release/wait event while the O6 action is being applied.
        if hand_command is not None:
            return reference_index

        action_index = min(
            reference_index + self.lookahead_frames,
            self.num_frames - 1,
        )

        # Never look past the next unfired hand event. The arm may approach the
        # event frame quickly, but the event itself is still triggered only when
        # the observation-aligned cursor reaches it.
        next_event_frame = self._next_pending_event_frame(episode_id)
        if next_event_frame is not None:
            action_index = min(action_index, next_event_frame)
        return max(reference_index, action_index)

    def align(self, episode_id: str, current_arm: np.ndarray) -> AlignmentResult:
        current = np.asarray(current_arm, dtype=np.float32)
        if current.shape != (ARM_DIM,) or not np.isfinite(current).all():
            raise ValueError(f"current arm state must contain {ARM_DIM} finite values")
        with self._lock:
            index, distance = self._nearest(episode_id, current)
            hand_command, event_id = self._next_due_event(episode_id, index)
            action_index = self._action_reference_index(
                episode_id,
                index,
                hand_command,
            )
            fired = self._fired.setdefault(episode_id, set())
            done = bool(index >= self.num_frames - 1 and len(fired) == len(self.events))
            return AlignmentResult(
                reference_index=index,
                action_reference_index=action_index,
                match_distance=distance,
                target_arm=self.reference[action_index].astype(np.float32, copy=True),
                hand_command=hand_command,
                hand_event_id=event_id,
                done=done,
            )

    def reset(self, episode_id: str | None) -> None:
        with self._lock:
            if episode_id is None:
                self._cursor.clear()
                self._fired.clear()
            else:
                self._cursor.pop(episode_id, None)
                self._fired.pop(episode_id, None)
