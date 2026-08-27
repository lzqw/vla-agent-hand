"""Rabo SDK device wrapper for remote VLA/oracle execution."""

from __future__ import annotations

import concurrent.futures
import threading
import time
from collections.abc import Sequence

import numpy as np

from .rabo_config import CollectorConfig


class RaboDevices:
    def __init__(self, config: CollectorConfig, mode: str = "sim") -> None:
        self.config = config
        try:
            from rabo_robocap import LinkerArmA7, LinkerHandO6Left, LinkerHandO6Right
        except ImportError as exc:
            raise RuntimeError("rabo_robocap is required in the Rabo runtime") from exc

        rabo = config.rabo
        # Fixed-oracle structured commands do not require joint-limit metadata.
        # Keep the exported SDK classes and build numeric-action specs lazily only
        # if a future 26D VLA action is actually enabled.
        self._arm_class = LinkerArmA7
        self._left_hand_class = LinkerHandO6Left
        self._right_hand_class = LinkerHandO6Right
        self._arm_limits: np.ndarray | None = None
        self._left_hand_limits: np.ndarray | None = None
        self._left_hand_drive: np.ndarray | None = None
        self._right_hand_limits: np.ndarray | None = None
        self._right_hand_drive: np.ndarray | None = None

        self.left_arm = LinkerArmA7(robot_id=rabo["left_arm_id"], mode=mode)
        self.right_arm = LinkerArmA7(robot_id=rabo["right_arm_id"], mode=mode)
        self.left_hand = LinkerHandO6Left(robot_id=rabo["left_hand_id"], mode=mode)
        self.right_hand = LinkerHandO6Right(robot_id=rabo["right_hand_id"], mode=mode)
        self._command_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="rabo-command"
        )

    @staticmethod
    def _valid_part(values: object, expected: int) -> bool:
        return isinstance(values, Sequence) and len(values) == expected

    def read_full_state(self) -> np.ndarray | None:
        parts = (
            self.left_arm.get_joint_angles(),
            self.right_arm.get_joint_angles(),
            self.left_hand.get_joint_angles(),
            self.right_hand.get_joint_angles(),
        )
        expected = (7, 7, 11, 11)
        if any(not self._valid_part(part, size) for part, size in zip(parts, expected)):
            return None
        values = np.asarray(
            [float(value) for part in parts for value in part], dtype=np.float32
        )
        if values.shape != (36,) or not np.isfinite(values).all():
            return None
        return values

    @staticmethod
    def _joint_limit_spec(robot_class: object, dof: int) -> np.ndarray:
        limits = np.asarray(getattr(robot_class, "JOINT_LIMITS", None), dtype=np.float32)
        if limits.shape != (dof, 2) or not np.isfinite(limits).all():
            raise RuntimeError(f"Invalid JOINT_LIMITS on {robot_class}: {limits.shape}")
        limits = limits.copy()
        limits[:, 0] += np.float32(0.002)
        limits[:, 1] -= np.float32(0.002)
        return limits

    @staticmethod
    def _hand_spec(hand_class: object) -> tuple[np.ndarray, np.ndarray]:
        limits = np.asarray(getattr(hand_class, "JOINT_LIMITS", None), dtype=np.float32)
        drive = np.asarray(getattr(hand_class, "CLENCH_DRIVE", None), dtype=np.int64)
        expected_drive = np.asarray((0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5), dtype=np.int64)
        if limits.shape != (11, 2) or drive.shape != (11,) or not np.array_equal(drive, expected_drive):
            raise RuntimeError(f"Unexpected hand spec on {hand_class}")
        limits = limits.copy()
        limits[:, 1] -= np.float32(0.002)
        return limits, drive

    def _ensure_numeric_specs(self) -> None:
        if self._arm_limits is not None:
            return
        self._arm_limits = self._joint_limit_spec(self._arm_class, 7)
        self._left_hand_limits, self._left_hand_drive = self._hand_spec(self._left_hand_class)
        self._right_hand_limits, self._right_hand_drive = self._hand_spec(self._right_hand_class)

    @staticmethod
    def _slew(desired: np.ndarray, current: np.ndarray, max_step: float) -> np.ndarray:
        return (current + np.clip(desired - current, -max_step, max_step)).astype(np.float32)

    @staticmethod
    def _expand_hand_target(
        active_target: np.ndarray,
        current_full: np.ndarray,
        limits: np.ndarray,
        drive: np.ndarray,
        max_step: float,
    ) -> np.ndarray:
        active_indices = np.asarray((0, 1, 3, 5, 7, 9), dtype=np.int64)
        active_limits = limits[active_indices]
        active_target = np.clip(active_target, active_limits[:, 0], active_limits[:, 1])
        span = active_limits[:, 1] - active_limits[:, 0]
        fraction = np.divide(
            active_target - active_limits[:, 0],
            span,
            out=np.zeros_like(active_target),
            where=span > 0,
        )
        full_target = limits[:, 0] + fraction[drive] * (limits[:, 1] - limits[:, 0])
        full_target = current_full + np.clip(full_target - current_full, -max_step, max_step)
        return np.clip(full_target, limits[:, 0], limits[:, 1]).astype(np.float32)

    def command_action(
        self,
        action: np.ndarray,
        current_full_state: np.ndarray,
        scope: str,
        max_arm_step: float,
        max_hand_step: float,
    ) -> dict[str, bool]:
        self._ensure_numeric_specs()
        assert self._arm_limits is not None
        assert self._left_hand_limits is not None and self._left_hand_drive is not None
        assert self._right_hand_limits is not None and self._right_hand_drive is not None

        action = np.asarray(action, dtype=np.float32)
        current = np.asarray(current_full_state, dtype=np.float32)
        if action.shape != (26,) or current.shape != (36,):
            raise ValueError(f"Invalid action/state shapes: {action.shape}, {current.shape}")

        left_arm = np.clip(
            self._slew(action[0:7], current[0:7], max_arm_step),
            self._arm_limits[:, 0], self._arm_limits[:, 1],
        )
        right_arm = np.clip(
            self._slew(action[7:14], current[7:14], max_arm_step),
            self._arm_limits[:, 0], self._arm_limits[:, 1],
        )
        commands: list[tuple[str, object, np.ndarray]] = [
            ("left_arm", self.left_arm, left_arm),
            ("right_arm", self.right_arm, right_arm),
        ]
        if scope == "full":
            left_hand = self._expand_hand_target(
                action[14:20], current[14:25], self._left_hand_limits,
                self._left_hand_drive, max_hand_step,
            )
            right_hand = self._expand_hand_target(
                action[20:26], current[25:36], self._right_hand_limits,
                self._right_hand_drive, max_hand_step,
            )
            commands.extend(
                (("left_hand", self.left_hand, left_hand),
                 ("right_hand", self.right_hand, right_hand))
            )

        futures = {
            name: self._command_executor.submit(
                device.move_joints, target.tolist(), blocking=False
            )
            for name, device, target in commands
        }
        results = {
            name: bool(future.result(timeout=2.0))
            for name, future in futures.items()
        }
        if not all(results.values()):
            self.stop()
            raise RuntimeError(f"SDK rejected command: {results}")
        return results

    @staticmethod
    def _move_joints_checked(arm, label: str, joints: list[float]) -> None:
        for attempt in range(2):
            result = arm.move_joints(joints)
            if result is not False:
                return
            print(f"[setup] {label} attempt {attempt + 1} returned False", flush=True)
            if attempt == 0:
                time.sleep(0.10)
        raise RuntimeError(f"{label}: move_joints returned False twice")

    def pre_position(self) -> None:
        self.right_hand.clench(0, 0, 0, 0, 0, 0)
        self.left_hand.clench(0, 0, 0, 0, 0, 0)
        errors: list[BaseException] = []
        lock = threading.Lock()

        def guarded(fn) -> None:
            try:
                fn()
            except BaseException as exc:
                with lock:
                    errors.append(exc)

        def right() -> None:
            self._move_joints_checked(
                self.right_arm, "right pre-position 1",
                [-1.57, -1.5, 0, -1.57, 0, -1, 0],
            )
            self._move_joints_checked(
                self.right_arm, "right pre-position 2",
                [0, 0, 0, -2, 0, 1, 0],
            )

        def left() -> None:
            self._move_joints_checked(
                self.left_arm, "left pre-position 1",
                [0, -1.57, 0, 0, 0, 0, 0],
            )
            self._move_joints_checked(
                self.left_arm, "left pre-position 2",
                [-1.57, -0.7, 0, 0, 0, 0, 0],
            )

        threads = [
            threading.Thread(target=guarded, args=(right,)),
            threading.Thread(target=guarded, args=(left,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise RuntimeError("pre_position failed") from errors[0]

    def reset_fixed_scene(self) -> None:
        try:
            from rabo_dev_kit import SetEntityPose
        except ImportError as exc:
            raise RuntimeError("rabo_dev_kit is required for fixed-scene reset") from exc
        fixed = {
            "B": (-0.3413, -0.1710, 0.2806, 0.0, 0.0, 0.5233),
            "A": (-0.2286, -0.0999, 0.2819, 0.0, 0.0, 0.5233),
            "C": (-0.2975, -0.0527, 0.2872, 0.0, 0.0, 0.5233),
        }
        setter = SetEntityPose(world=self.config.rabo["world_id"])
        for name, pose in fixed.items():
            setter.set(self.config.rabo["nut_ids"][name], pose)
        time.sleep(0.5)

    def stop(self) -> None:
        for device in (self.left_arm, self.right_arm, self.left_hand, self.right_hand):
            try:
                device.stop()
            except Exception as exc:
                print(f"[stop-warning] {type(device).__name__}: {exc!r}", flush=True)

    def shutdown(self) -> None:
        self._command_executor.shutdown(wait=True, cancel_futures=True)
        for device in (self.left_arm, self.right_arm, self.left_hand, self.right_hand):
            try:
                device.shutdown()
            except Exception as exc:
                print(f"[shutdown-warning] {type(device).__name__}: {exc!r}", flush=True)
