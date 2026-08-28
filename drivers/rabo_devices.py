"""Rabo SDK device wrapper used by the VLA runtime."""

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
        self._arm_class = LinkerArmA7
        self._arm_limits: np.ndarray | None = None

        self.left_arm = LinkerArmA7(robot_id=rabo["left_arm_id"], mode=mode)
        self.right_arm = LinkerArmA7(robot_id=rabo["right_arm_id"], mode=mode)
        self.left_hand = LinkerHandO6Left(robot_id=rabo["left_hand_id"], mode=mode)
        self.right_hand = LinkerHandO6Right(robot_id=rabo["right_hand_id"], mode=mode)
        self._joint_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="rabo-vla-joint"
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

    def _ensure_arm_limits(self) -> None:
        if self._arm_limits is None:
            self._arm_limits = self._joint_limit_spec(self._arm_class, 7)

    @staticmethod
    def _slew(desired: np.ndarray, current: np.ndarray, max_step: float) -> np.ndarray:
        return (current + np.clip(desired - current, -max_step, max_step)).astype(np.float32)

    def _submit_arm_targets(
        self, left_target: np.ndarray, right_target: np.ndarray
    ) -> dict[str, bool]:
        futures = {
            "left_arm": self._joint_executor.submit(
                self.left_arm.move_joints, left_target.tolist(), blocking=False
            ),
            "right_arm": self._joint_executor.submit(
                self.right_arm.move_joints, right_target.tolist(), blocking=False
            ),
        }
        results = {
            name: bool(future.result(timeout=2.0))
            for name, future in futures.items()
        }
        if not all(results.values()):
            self.stop()
            raise RuntimeError(f"SDK rejected arm action: {results}")
        return results

    def command_arm_joint_action(
        self,
        action: np.ndarray,
        current_full_state: np.ndarray,
        *,
        max_arm_step: float,
    ) -> dict[str, bool]:
        """Execute one 14D absolute joint-position action for the two A7 arms."""
        self._ensure_arm_limits()
        assert self._arm_limits is not None

        action = np.asarray(action, dtype=np.float32)
        current = np.asarray(current_full_state, dtype=np.float32)
        if action.shape != (14,) or current.shape != (36,):
            raise ValueError(f"Invalid arm action/state shapes: {action.shape}, {current.shape}")
        if not np.isfinite(action).all() or not np.isfinite(current).all():
            raise ValueError("arm action/state contains non-finite values")

        left_target = np.clip(
            self._slew(action[0:7], current[0:7], max_arm_step),
            self._arm_limits[:, 0],
            self._arm_limits[:, 1],
        )
        right_target = np.clip(
            self._slew(action[7:14], current[7:14], max_arm_step),
            self._arm_limits[:, 0],
            self._arm_limits[:, 1],
        )
        return self._submit_arm_targets(left_target, right_target)

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
                self.right_arm,
                "right pre-position 1",
                [-1.57, -1.5, 0, -1.57, 0, -1, 0],
            )
            self._move_joints_checked(
                self.right_arm,
                "right pre-position 2",
                [0, 0, 0, -2, 0, 1, 0],
            )

        def left() -> None:
            self._move_joints_checked(
                self.left_arm,
                "left pre-position 1",
                [0, -1.57, 0, 0, 0, 0, 0],
            )
            self._move_joints_checked(
                self.left_arm,
                "left pre-position 2",
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
        self._joint_executor.shutdown(wait=True, cancel_futures=True)
        for device in (self.left_arm, self.right_arm, self.left_hand, self.right_hand):
            try:
                device.shutdown()
            except Exception as exc:
                print(f"[shutdown-warning] {type(device).__name__}: {exc!r}", flush=True)
