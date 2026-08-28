"""Dense arm alignment with robust one-shot O6 hand-event scheduling.

The 4080 policy aligns the live 14D A7 arm state to a successful dense
trajectory. Arm targets may use a short lookahead, but the alignment cursor is
never allowed to cross an unfired hand event. Hand events are additionally
gated by actual arm convergence for consecutive control cycles before
clench/grasp-force/wait is emitted. Grasp-force events can be repeated while
the arm is held at the grasp pose.
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
    hand_event_frame: int | None
    hand_event_error_rad: float | None
    hand_event_settle_count: int
    hand_event_repeat_index: int
    done: bool


class ArmHandReference:
    """Align live arm state to a dense trajectory and safely schedule O6 actions."""

    def __init__(
        self,
        reference_path: Path,
        hand_events_path: Path,
        *,
        initial_search: int = 250,
        forward_window: int = 80,
        target_tolerance_rad: float = 0.025,
        lookahead_frames: int = 3,
        hand_event_tolerance_rad: float = 0.020,
        hand_event_settle_cycles: int = 2,
        grasp_force_repeats: int = 2,
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
        if not np.isfinite(hand_event_tolerance_rad) or hand_event_tolerance_rad <= 0.0:
            raise ValueError("hand_event_tolerance_rad must be positive and finite")
        if int(hand_event_settle_cycles) < 1:
            raise ValueError("hand_event_settle_cycles must be >= 1")
        if int(grasp_force_repeats) < 1:
            raise ValueError("grasp_force_repeats must be >= 1")

        self.target_tolerance_rad = float(target_tolerance_rad)
        self.lookahead_frames = max(1, int(lookahead_frames))
        self.hand_event_tolerance_rad = float(hand_event_tolerance_rad)
        self.hand_event_settle_cycles = int(hand_event_settle_cycles)
        self.grasp_force_repeats = int(grasp_force_repeats)
        self.scale = np.maximum(np.std(reference, axis=0), np.float32(0.03)).astype(
            np.float32
        )

        self._cursor: dict[str, int] = {}
        self._fired: dict[str, set[int]] = {}
        self._settle: dict[str, dict[int, int]] = {}
        self._repeat: dict[str, dict[int, int]] = {}
        self._lock = threading.Lock()

    def _next_pending_event(self, episode_id: str) -> dict[str, Any] | None:
        fired = self._fired.setdefault(episode_id, set())
        for event in self.events:
            if int(event["event_id"]) not in fired:
                return event
        return None

    def _nearest(self, episode_id: str, current_arm: np.ndarray) -> tuple[int, float]:
        previous = self._cursor.get(episode_id, -1)
        if previous < 0:
            start = 0
            end = self.initial_search
        else:
            start = max(0, previous - 2)
            end = min(self.num_frames, previous + self.forward_window + 1)

        # Critical monotonic event barrier: trajectory self-intersections can make
        # a much later frame look closer than the current grasp frame. Never let
        # nearest-neighbour alignment cross the next unfired hand event. This
        # avoids the old failure mode ref=93 -> target=64 -> ref=93 oscillation.
        pending = self._next_pending_event(episode_id)
        pending_frame = int(pending["frame_index"]) if pending is not None else None
        if pending_frame is not None and previous < pending_frame:
            end = min(end, pending_frame + 1)

        if end <= start:
            # Defensive fallback for a restored/corrupt cursor. Never command a
            # backwards jump; keep the current monotonic cursor instead.
            selected = max(0, min(previous, self.num_frames - 1))
            selected_delta = (self.reference[selected] - current_arm) / self.scale
            distance = float(np.mean(selected_delta * selected_delta))
            self._cursor[episode_id] = selected
            return selected, distance

        candidates = self.reference[start:end]
        normalized = (candidates - current_arm[None, :]) / self.scale[None, :]
        distances = np.mean(normalized * normalized, axis=1)
        local = int(np.argmin(distances))
        matched = start + local
        selected = max(previous, matched)

        if previous >= 0 and previous < self.num_frames - 1:
            target_error = float(np.max(np.abs(self.target[previous] - current_arm)))
            if target_error <= self.target_tolerance_rad:
                selected = max(selected, previous + 1)

        # The cursor itself may approach an unfired event but must not cross it.
        if pending_frame is not None:
            selected = min(selected, pending_frame)
        selected = min(selected, self.num_frames - 1)

        selected_delta = (self.reference[selected] - current_arm) / self.scale
        distance = float(np.mean(selected_delta * selected_delta))
        self._cursor[episode_id] = selected
        return selected, distance

    @staticmethod
    def _is_grasp_force(command: dict[str, Any]) -> bool:
        return str(command.get("action_type", "")).endswith("_hand_grasp_force")

    def _process_due_event(
        self,
        episode_id: str,
        reference_index: int,
        current_arm: np.ndarray,
    ) -> tuple[
        dict[str, Any] | None,
        int | None,
        int | None,
        float | None,
        int,
        int,
    ]:
        event = self._next_pending_event(episode_id)
        if event is None:
            return None, None, None, None, 0, 0

        event_id = int(event["event_id"])
        event_frame = int(event["frame_index"])
        if reference_index < event_frame:
            return None, event_id, event_frame, None, 0, 0

        event_error = float(np.max(np.abs(self.reference[event_frame] - current_arm)))
        settle = self._settle.setdefault(episode_id, {})
        repeats = self._repeat.setdefault(episode_id, {})

        if event_error > self.hand_event_tolerance_rad:
            settle[event_id] = 0
            return None, event_id, event_frame, event_error, 0, repeats.get(event_id, 0)

        settle_count = settle.get(event_id, 0) + 1
        settle[event_id] = settle_count
        if settle_count < self.hand_event_settle_cycles:
            return None, event_id, event_frame, event_error, settle_count, repeats.get(event_id, 0)

        command = dict(event["command"])
        repeat_goal = self.grasp_force_repeats if self._is_grasp_force(command) else 1
        repeat_index = repeats.get(event_id, 0) + 1
        repeats[event_id] = repeat_index

        if repeat_index >= repeat_goal:
            self._fired.setdefault(episode_id, set()).add(event_id)
            settle.pop(event_id, None)
            repeats.pop(event_id, None)

        return command, event_id, event_frame, event_error, settle_count, repeat_index

    def _action_reference_index(
        self,
        episode_id: str,
        reference_index: int,
        hand_command: dict[str, Any] | None,
        active_event_frame: int | None,
    ) -> int:
        # Hold at an active event frame only if it is not behind the monotonic
        # cursor. A server must never ask the robot to reverse along the dense
        # trajectory just to recover an old event.
        if (
            active_event_frame is not None
            and reference_index == active_event_frame
        ):
            return reference_index
        if hand_command is not None:
            return reference_index

        action_index = min(
            reference_index + self.lookahead_frames,
            self.num_frames - 1,
        )
        pending = self._next_pending_event(episode_id)
        if pending is not None:
            pending_frame = int(pending["frame_index"])
            if pending_frame >= reference_index:
                action_index = min(action_index, pending_frame)
        return max(reference_index, action_index)

    def align(self, episode_id: str, current_arm: np.ndarray) -> AlignmentResult:
        current = np.asarray(current_arm, dtype=np.float32)
        if current.shape != (ARM_DIM,) or not np.isfinite(current).all():
            raise ValueError(f"current arm state must contain {ARM_DIM} finite values")
        with self._lock:
            index, distance = self._nearest(episode_id, current)
            (
                hand_command,
                event_id,
                event_frame,
                event_error,
                settle_count,
                repeat_index,
            ) = self._process_due_event(episode_id, index, current)
            action_index = self._action_reference_index(
                episode_id,
                index,
                hand_command,
                event_frame,
            )
            # Hard invariant: remote arm targets are monotonic in reference time.
            if action_index < index:
                raise RuntimeError(
                    f"non-monotonic arm target: cursor={index}, target={action_index}"
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
                hand_event_frame=event_frame,
                hand_event_error_rad=event_error,
                hand_event_settle_count=settle_count,
                hand_event_repeat_index=repeat_index,
                done=done,
            )

    def reset(self, episode_id: str | None) -> None:
        with self._lock:
            if episode_id is None:
                self._cursor.clear()
                self._fired.clear()
                self._settle.clear()
                self._repeat.clear()
            else:
                self._cursor.pop(episode_id, None)
                self._fired.pop(episode_id, None)
                self._settle.pop(episode_id, None)
                self._repeat.pop(episode_id, None)
