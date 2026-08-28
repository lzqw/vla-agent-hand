"""Dense arm alignment with stable event-aware target latching.

The 4080 policy aligns the live 14D A7 arm state to a successful dense
trajectory. The alignment cursor cannot cross an unfired hand event, and the
arm target itself is latched until the robot has converged to it. This avoids
chasing a new joint target on every network cycle while an A7 command is still
in flight. Hand events are gated by actual arm convergence for consecutive
control cycles before clench/grasp-force/wait is emitted.
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
    action_target_error_rad: float | None
    action_target_latched: bool
    hand_command: dict[str, Any] | None
    hand_event_id: int | None
    hand_event_frame: int | None
    hand_event_error_rad: float | None
    hand_event_settle_count: int
    hand_event_repeat_index: int
    done: bool


class ArmHandReference:
    """Align live arm state and emit stable monotonic arm/hand targets."""

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
        action_retarget_tolerance_rad: float = 0.030,
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

        for name, value in (
            ("target_tolerance_rad", target_tolerance_rad),
            ("hand_event_tolerance_rad", hand_event_tolerance_rad),
            ("action_retarget_tolerance_rad", action_retarget_tolerance_rad),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if int(hand_event_settle_cycles) < 1:
            raise ValueError("hand_event_settle_cycles must be >= 1")
        if int(grasp_force_repeats) < 1:
            raise ValueError("grasp_force_repeats must be >= 1")

        self.target_tolerance_rad = float(target_tolerance_rad)
        self.lookahead_frames = max(1, int(lookahead_frames))
        self.hand_event_tolerance_rad = float(hand_event_tolerance_rad)
        self.hand_event_settle_cycles = int(hand_event_settle_cycles)
        self.grasp_force_repeats = int(grasp_force_repeats)
        self.action_retarget_tolerance_rad = float(action_retarget_tolerance_rad)
        self.scale = np.maximum(np.std(reference, axis=0), np.float32(0.03)).astype(
            np.float32
        )

        self._cursor: dict[str, int] = {}
        self._action_cursor: dict[str, int] = {}
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

        pending = self._next_pending_event(episode_id)
        pending_frame = int(pending["frame_index"]) if pending is not None else None
        if pending_frame is not None and previous < pending_frame:
            end = min(end, pending_frame + 1)

        if end <= start:
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

    def _desired_action_reference_index(
        self,
        episode_id: str,
        reference_index: int,
        hand_command: dict[str, Any] | None,
        active_event_frame: int | None,
    ) -> int:
        if active_event_frame is not None and reference_index == active_event_frame:
            return reference_index
        if hand_command is not None:
            return reference_index

        action_index = min(reference_index + self.lookahead_frames, self.num_frames - 1)
        pending = self._next_pending_event(episode_id)
        if pending is not None:
            pending_frame = int(pending["frame_index"])
            if pending_frame >= reference_index:
                action_index = min(action_index, pending_frame)
        return max(reference_index, action_index)

    def _latched_action_reference_index(
        self,
        episode_id: str,
        desired_index: int,
        current_arm: np.ndarray,
    ) -> tuple[int, float | None, bool]:
        """Hold one target until the physical arm reaches it.

        The web executor sends every returned action to the A7 SDK. Updating the
        target at network rate can leave the SDK handling multiple in-flight
        commands and creates visible endpoint chatter. A per-episode target
        latch makes repeated observations return the exact same target until
        the prior one is reached.
        """
        previous_target = self._action_cursor.get(episode_id)
        if previous_target is None:
            selected = int(desired_index)
            self._action_cursor[episode_id] = selected
            return selected, None, False

        previous_error = float(
            np.max(np.abs(self.reference[previous_target] - current_arm))
        )
        if previous_error > self.action_retarget_tolerance_rad:
            return previous_target, previous_error, True

        selected = max(previous_target, int(desired_index))
        pending = self._next_pending_event(episode_id)
        if pending is not None:
            pending_frame = int(pending["frame_index"])
            if previous_target <= pending_frame:
                selected = min(selected, pending_frame)
        selected = max(previous_target, selected)
        self._action_cursor[episode_id] = selected
        return selected, previous_error, False

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
            desired_index = self._desired_action_reference_index(
                episode_id,
                index,
                hand_command,
                event_frame,
            )
            action_index, action_error, action_latched = self._latched_action_reference_index(
                episode_id,
                desired_index,
                current,
            )
            if action_index < index:
                # A lookahead target may legitimately be ahead of the alignment
                # cursor, but it must never fall behind it. If the observation
                # catches up unexpectedly, advance the latch rather than reverse.
                action_index = index
                self._action_cursor[episode_id] = action_index
                action_latched = False

            fired = self._fired.setdefault(episode_id, set())
            done = bool(index >= self.num_frames - 1 and len(fired) == len(self.events))
            return AlignmentResult(
                reference_index=index,
                action_reference_index=action_index,
                match_distance=distance,
                target_arm=self.reference[action_index].astype(np.float32, copy=True),
                action_target_error_rad=action_error,
                action_target_latched=action_latched,
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
                self._action_cursor.clear()
                self._fired.clear()
                self._settle.clear()
                self._repeat.clear()
            else:
                self._cursor.pop(episode_id, None)
                self._action_cursor.pop(episode_id, None)
                self._fired.pop(episode_id, None)
                self._settle.pop(episode_id, None)
                self._repeat.pop(episode_id, None)
