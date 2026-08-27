"""Three-camera ROS2 sensing plus SDK joint-state sampling for the VLA agent."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .rabo_config import CollectorConfig


@dataclass(frozen=True)
class SensorSnapshot:
    sim_time: float
    monotonic_time: float
    state: np.ndarray
    full_state: np.ndarray
    images: dict[str, np.ndarray]
    joint_ages: dict[str, float]
    image_stamps: dict[str, float]
    camera_reused: dict[str, bool]


@dataclass(frozen=True)
class _Entry:
    seq: int
    stamp: float
    arrival_time: float
    value: Any


class _Ring:
    def __init__(self, depth: int) -> None:
        self._data: deque[_Entry] = deque(maxlen=max(2, depth))
        self._seq = 0

    def append(self, stamp: float, value: Any) -> None:
        self._seq += 1
        self._data.append(_Entry(self._seq, stamp, time.monotonic(), value))

    def newest(self) -> _Entry | None:
        return self._data[-1] if self._data else None


def header_stamp_seconds(msg: Any) -> float:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return 0.0
    return float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0)) * 1e-9


def image_to_bgr8(msg: Any) -> np.ndarray:
    height = int(msg.height)
    width = int(msg.width)
    encoding = str(msg.encoding).lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if encoding in {"bgr8", "8uc3"}:
        image = raw.reshape(height, int(msg.step))[:, : width * 3].reshape(height, width, 3)
        return image.copy()
    if encoding == "rgb8":
        image = raw.reshape(height, int(msg.step))[:, : width * 3].reshape(height, width, 3)
        return image[:, :, ::-1].copy()
    if encoding in {"mono8", "8uc1"}:
        gray = raw.reshape(height, int(msg.step))[:, :width]
        return np.repeat(gray[:, :, None], 3, axis=2).copy()
    raise RuntimeError(f"unsupported image encoding: {msg.encoding!r}")


class Ros2SensorBackend:
    """Subscribe only to RGB cameras; read 36D joints through the official SDK."""

    def __init__(
        self,
        config: CollectorConfig,
        joint_reader: Callable[[], np.ndarray | None] | None = None,
        abort_event: threading.Event | None = None,
    ) -> None:
        self.config = config
        self._joint_reader = joint_reader
        self._abort_event = abort_event or threading.Event()
        self._cond = threading.Condition()
        self._camera_rings = {
            name: _Ring(config.ring_buffer_depth) for name in config.camera_topics
        }
        self._camera_msg_count = {name: 0 for name in config.camera_topics}
        self._camera_last_arrival = {name: 0.0 for name in config.camera_topics}
        self._last_camera_seq = {name: -1 for name in config.camera_topics}
        self._camera_node = None
        self._camera_executor = None
        self._camera_thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._episode_wall_t0 = 0.0
        self._next_sample_wall = 0.0
        self._sample_index = 0

    def start(self) -> None:
        if self._started:
            return
        try:
            import rclpy
            from rclpy.executors import MultiThreadedExecutor
            from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise RuntimeError("ROS2 sensing requires rclpy and sensor_msgs") from exc

        if not rclpy.ok():
            rclpy.init(args=None)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._camera_node = rclpy.create_node("vla_agent_camera_reader")
        for name, topic in self.config.camera_topics.items():
            self._camera_node.create_subscription(
                Image,
                topic,
                lambda msg, camera_name=name: self._on_image(camera_name, msg),
                qos,
            )
        self._camera_executor = MultiThreadedExecutor(num_threads=2)
        self._camera_executor.add_node(self._camera_node)
        self._camera_thread = threading.Thread(
            target=self._camera_executor.spin,
            name="vla-agent-cameras",
            daemon=True,
        )
        self._camera_thread.start()
        self._started = True

    def _on_image(self, name: str, msg: Any) -> None:
        with self._cond:
            self._camera_rings[name].append(header_stamp_seconds(msg), msg)
            self._camera_msg_count[name] += 1
            self._camera_last_arrival[name] = time.monotonic()
            self._cond.notify_all()

    def wait_ready(self, timeout_s: float) -> None:
        need = int(self.config.dataset.get("camera_ready_min_frames", 5))
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while True:
                missing = [name for name, count in self._camera_msg_count.items() if count < need]
                if not missing:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"camera readiness timeout: {missing}")
                self._cond.wait(min(remaining, 0.5))

    def reset_clock(self) -> None:
        now = time.monotonic()
        self._episode_wall_t0 = now
        self._next_sample_wall = now
        self._sample_index = 0
        self._last_camera_seq = {name: -1 for name in self.config.camera_topics}

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def _read_joints(self) -> np.ndarray | None:
        if self._joint_reader is None:
            return None
        values = self._joint_reader()
        if values is None:
            return None
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (36,) or not np.isfinite(values).all():
            return None
        return values

    def sample_next(self, dt: float, timeout_s: float, stop_event: threading.Event) -> SensorSnapshot | None:
        del timeout_s
        remaining = self._next_sample_wall - time.monotonic()
        if remaining > 0 and stop_event.wait(remaining):
            return None
        if stop_event.is_set():
            return None

        now = time.monotonic()
        stall_timeout = float(self.config.dataset.get("camera_stall_timeout_s", 3.0))
        with self._cond:
            entries = {name: ring.newest() for name, ring in self._camera_rings.items()}
            for name, last in self._camera_last_arrival.items():
                if last > 0 and now - last > stall_timeout:
                    raise RuntimeError(f"camera {name} stalled for >{stall_timeout}s")
        if any(entry is None for entry in entries.values()):
            return None

        full_state = self._read_joints()
        if full_state is None:
            return None
        by_name = dict(zip(self.config.full_state_names, full_state))
        state = np.asarray([by_name[name] for name in self.config.state_names], dtype=np.float32)

        images: dict[str, np.ndarray] = {}
        image_stamps: dict[str, float] = {}
        camera_reused: dict[str, bool] = {}
        for name, entry in entries.items():
            assert entry is not None
            images[name] = image_to_bgr8(entry.value)
            image_stamps[name] = float(entry.stamp)
            camera_reused[name] = self._last_camera_seq[name] == entry.seq
            self._last_camera_seq[name] = entry.seq

        sim_time = self._sample_index * dt
        self._sample_index += 1
        self._next_sample_wall = max(self._next_sample_wall + dt, now)
        return SensorSnapshot(
            sim_time=sim_time,
            monotonic_time=now,
            state=state,
            full_state=full_state,
            images=images,
            joint_ages={name: 0.0 for name in self.config.full_state_names},
            image_stamps=image_stamps,
            camera_reused=camera_reused,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.wake()
        if self._camera_executor is not None:
            try:
                self._camera_executor.shutdown(timeout_sec=3.0)
            except TypeError:
                self._camera_executor.shutdown()
            except Exception:
                pass
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=3.0)
        if self._camera_node is not None:
            try:
                self._camera_node.destroy_node()
            except Exception:
                pass


class MockSensorBackend:
    def __init__(self, config: CollectorConfig) -> None:
        self.config = config
        self._index = 0
        self._next = 0.0

    def start(self) -> None:
        self.reset_clock()

    def wait_ready(self, timeout_s: float) -> None:
        del timeout_s

    def reset_clock(self) -> None:
        self._index = 0
        self._next = time.monotonic()

    def wake(self) -> None:
        pass

    def sample_next(self, dt: float, timeout_s: float, stop_event: threading.Event) -> SensorSnapshot | None:
        del timeout_s
        remaining = self._next - time.monotonic()
        if remaining > 0 and stop_event.wait(remaining):
            return None
        idx = self._index
        self._index += 1
        self._next += dt
        full = np.zeros(36, dtype=np.float32)
        state = np.zeros(26, dtype=np.float32)
        images = {
            name: np.full((270, 480, 3), (idx * 13) % 255, dtype=np.uint8)
            for name in self.config.camera_topics
        }
        return SensorSnapshot(
            sim_time=idx * dt,
            monotonic_time=time.monotonic(),
            state=state,
            full_state=full,
            images=images,
            joint_ages={},
            image_stamps={name: idx * dt for name in images},
            camera_reused={name: False for name in images},
        )

    def close(self) -> None:
        pass
