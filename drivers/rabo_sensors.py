"""Three-camera ROS2 sensing plus SDK joint-state sampling for the VLA agent.

This module intentionally mirrors the camera configuration that was already
validated by the earlier simple_v1 collector on the Rabo platform:

- camera-only ROS subscriptions (no 36 high-rate joint subscriptions)
- RELIABLE / VOLATILE QoS with depth 6
- one mutually-exclusive callback group per camera
- a 2-thread executor (single-thread fallback)
- wall-clock sampling of the newest cached image
- 480x270 center-crop/resize according to config.yaml

The primary camera must remain fresh. Wrist cameras may temporarily reuse their
latest cached frame; that condition is exposed through ``camera_reused`` and a
rate-limited warning rather than aborting the episode.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

import cv2
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
    source_stamp: float
    arrival_time: float
    value: Any


class _Ring:
    def __init__(self, depth: int) -> None:
        self._data: deque[_Entry] = deque(maxlen=max(2, int(depth)))
        self._seq = 0

    def append(self, source_stamp: float, value: Any) -> None:
        self._seq += 1
        self._data.append(
            _Entry(
                seq=self._seq,
                source_stamp=float(source_stamp),
                arrival_time=time.monotonic(),
                value=value,
            )
        )

    def newest(self) -> _Entry | None:
        return self._data[-1] if self._data else None


def header_stamp_seconds(msg: Any) -> float:
    """Read ROS header time; zero/missing stamps fall back to arrival wall clock."""
    try:
        stamp = msg.header.stamp
        sec = int(stamp.sec)
        nanosec = int(stamp.nanosec)
    except (AttributeError, TypeError, ValueError):
        return time.monotonic()
    if sec == 0 and nanosec == 0:
        return time.monotonic()
    return float(sec) + float(nanosec) * 1e-9


def image_to_bgr8(msg: Any) -> np.ndarray:
    """Decode sensor_msgs/Image without cv_bridge."""
    encoding = str(msg.encoding).lower()
    layouts = {
        "bgr8": 3,
        "8uc3": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
        "8uc1": 1,
    }
    if encoding not in layouts:
        raise RuntimeError(f"unsupported image encoding: {msg.encoding!r}")

    channels = layouts[encoding]
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    row_bytes = width * channels
    if step < row_bytes:
        raise RuntimeError(f"Image.step={step} is smaller than row bytes={row_bytes}")

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    required = height * step
    if raw.size < required:
        raise RuntimeError(f"Image.data={raw.size} is smaller than height*step={required}")

    rows = raw[:required].reshape(height, step)
    pixels = rows[:, :row_bytes].reshape(height, width, channels)
    if encoding == "rgb8":
        pixels = pixels[..., ::-1]
    elif encoding == "rgba8":
        pixels = pixels[..., [2, 1, 0]]
    elif encoding == "bgra8":
        pixels = pixels[..., :3]
    elif encoding in {"mono8", "8uc1"}:
        pixels = np.repeat(pixels, 3, axis=2)
    return np.ascontiguousarray(pixels, dtype=np.uint8).copy()


def center_crop_resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Center crop to target aspect ratio, then resize without geometric stretch."""
    source_h, source_w = frame.shape[:2]
    source_ratio = source_w / source_h
    target_ratio = width / height
    if source_ratio > target_ratio:
        crop_w = max(1, int(round(source_h * target_ratio)))
        left = (source_w - crop_w) // 2
        frame = frame[:, left:left + crop_w]
    elif source_ratio < target_ratio:
        crop_h = max(1, int(round(source_w / target_ratio)))
        top = (source_h - crop_h) // 2
        frame = frame[top:top + crop_h, :]
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame, dtype=np.uint8)


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
        depth = max(2, int(config.ring_buffer_depth))
        self._camera_rings = {name: _Ring(depth) for name in config.camera_topics}
        self._camera_msg_count = {name: 0 for name in config.camera_topics}
        self._camera_last_arrival = {name: 0.0 for name in config.camera_topics}
        self._last_camera_seq = {name: -1 for name in config.camera_topics}
        self._last_stale_warning = {name: 0.0 for name in config.camera_topics}

        self._camera_node = None
        self._camera_executor = None
        self._camera_thread: threading.Thread | None = None
        self._started = False
        self._closed = False

        self._clock_lock = threading.Lock()
        self._episode_wall_t0 = 0.0
        self._next_sample_wall = 0.0
        self._sample_index = 0

    def start(self) -> None:
        if self._started:
            return
        try:
            import rclpy
            from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
            from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise RuntimeError("ROS2 sensing requires rclpy and sensor_msgs") from exc

        if not rclpy.ok():
            rclpy.init(args=None)

        # Proven settings from dexora_rabo_collector_simple_v1.
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=6,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._camera_node = rclpy.create_node("vla_agent_camera_reader")
        for name, topic in self.config.camera_topics.items():
            group = MutuallyExclusiveCallbackGroup()
            self._camera_node.create_subscription(
                Image,
                topic,
                lambda msg, camera_name=name: self._on_image(camera_name, msg),
                qos,
                callback_group=group,
            )

        try:
            self._camera_executor = MultiThreadedExecutor(num_threads=2)
        except Exception:
            self._camera_executor = SingleThreadedExecutor()
        self._camera_executor.add_node(self._camera_node)
        self._camera_thread = threading.Thread(
            target=self._camera_executor.spin,
            name="vla-agent-cameras",
            daemon=True,
        )
        self._camera_thread.start()
        self._started = True
        print(
            "[camera] ROS2 subscriptions started: QoS=RELIABLE depth=6 "
            f"target={int(self.config.dataset['image_width'])}x"
            f"{int(self.config.dataset['image_height'])}",
            flush=True,
        )

    def _on_image(self, name: str, msg: Any) -> None:
        with self._cond:
            self._camera_rings[name].append(header_stamp_seconds(msg), msg)
            self._camera_msg_count[name] += 1
            self._camera_last_arrival[name] = time.monotonic()
            self._cond.notify_all()

    def wait_ready(self, timeout_s: float) -> None:
        primary = self.config.primary_camera
        primary_need = int(self.config.dataset.get("camera_ready_min_frames", 5))
        deadline = time.monotonic() + timeout_s
        next_report = 0.0
        with self._cond:
            while True:
                missing: list[str] = []
                for name, count in self._camera_msg_count.items():
                    # Keep the high camera strict. Wrist cameras only need an initial frame;
                    # they may subsequently reuse their newest cached frame.
                    need = primary_need if name == primary else 1
                    if count < need:
                        missing.append(f"{name}({count}/{need})")
                if not missing:
                    print(
                        "[camera] ready counts="
                        + ", ".join(
                            f"{name}:{count}" for name, count in self._camera_msg_count.items()
                        ),
                        flush=True,
                    )
                    return

                now = time.monotonic()
                if now >= next_report:
                    print(
                        "[camera] waiting counts="
                        + ", ".join(
                            f"{name}:{count}" for name, count in self._camera_msg_count.items()
                        )
                        + f" missing={missing}",
                        flush=True,
                    )
                    next_report = now + 2.0

                remaining = deadline - now
                if remaining <= 0:
                    raise TimeoutError(
                        "camera readiness timeout: "
                        f"{missing}; counts={self._camera_msg_count}; "
                        "expected proven Rabo camera QoS RELIABLE/depth6"
                    )
                self._cond.wait(min(remaining, 0.5))

    def reset_clock(self) -> None:
        now = time.monotonic()
        with self._clock_lock:
            self._episode_wall_t0 = now
            self._next_sample_wall = now
            self._sample_index = 0
            self._last_camera_seq = {name: -1 for name in self.config.camera_topics}

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()

    @staticmethod
    def _wait_for_tick(target: float, stop_event: threading.Event) -> bool:
        while True:
            if stop_event.is_set():
                return False
            remaining = target - time.monotonic()
            if remaining <= 0:
                return True
            stop_event.wait(min(remaining, 0.02))

    def _read_joints(self) -> np.ndarray | None:
        if self._joint_reader is None:
            return None
        try:
            values = self._joint_reader()
        except Exception as exc:
            raise RuntimeError("failed to read joint state through Rabo SDK") from exc
        if values is None:
            return None
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (36,) or not np.isfinite(values).all():
            return None
        return values

    def sample_next(
        self,
        dt: float,
        timeout_s: float,
        stop_event: threading.Event,
    ) -> SensorSnapshot | None:
        del timeout_s  # sample newest cache at the requested wall-clock cadence
        with self._clock_lock:
            target = self._next_sample_wall
        if not self._wait_for_tick(target, stop_event):
            return None

        now = time.monotonic()
        stall_timeout = float(self.config.dataset.get("camera_stall_timeout_s", 3.0))
        primary = self.config.primary_camera

        with self._cond:
            entries = {name: ring.newest() for name, ring in self._camera_rings.items()}
            primary_last = self._camera_last_arrival.get(primary, 0.0)
            if primary_last <= 0.0 or now - primary_last > stall_timeout:
                self._abort_event.set()
                raise RuntimeError(
                    f"primary camera {primary} stalled for >{stall_timeout:.1f}s"
                )
            for name, last in self._camera_last_arrival.items():
                if name == primary or last <= 0.0:
                    continue
                age = now - last
                if age > stall_timeout and now - self._last_stale_warning[name] > 5.0:
                    print(
                        f"[camera-warning] secondary camera {name} stale for {age:.1f}s; "
                        "reusing latest cached frame",
                        flush=True,
                    )
                    self._last_stale_warning[name] = now

        if any(entry is None for entry in entries.values()):
            return None

        full_state = self._read_joints()
        if full_state is None:
            return None
        by_name = dict(zip(self.config.full_state_names, full_state))
        state = np.asarray([by_name[name] for name in self.config.state_names], dtype=np.float32)

        with self._clock_lock:
            index = self._sample_index
            self._sample_index += 1
            sim_time = index * max(0.0, dt)
            self._next_sample_wall += max(0.0, dt)
            # Do not burst catch-up samples if commands/JPEG/network took too long.
            if dt > 0.0 and self._next_sample_wall < now - dt:
                self._next_sample_wall = now + dt
            elif dt <= 0.0:
                self._next_sample_wall = now

        target_w = int(self.config.dataset["image_width"])
        target_h = int(self.config.dataset["image_height"])
        images: dict[str, np.ndarray] = {}
        image_stamps: dict[str, float] = {}
        camera_reused: dict[str, bool] = {}
        for name, entry in entries.items():
            assert entry is not None
            try:
                decoded = image_to_bgr8(entry.value)
                images[name] = center_crop_resize(decoded, target_w, target_h)
            except Exception as exc:
                raise RuntimeError(f"failed to decode/resize camera {name}") from exc
            camera_reused[name] = self._last_camera_seq[name] == entry.seq
            self._last_camera_seq[name] = entry.seq
            age = max(0.0, now - entry.arrival_time)
            image_stamps[name] = sim_time - age

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

    @staticmethod
    def _shutdown_executor(executor: Any) -> None:
        try:
            executor.shutdown(timeout_sec=5.0)
        except TypeError:
            executor.shutdown()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.wake()

        if self._camera_executor is not None:
            try:
                self._shutdown_executor(self._camera_executor)
            except Exception as exc:
                print(f"[camera-cleanup] executor shutdown warning: {exc!r}", flush=True)

        if self._camera_thread is not None:
            self._camera_thread.join(timeout=5.0)

        if self._camera_thread is not None and self._camera_thread.is_alive():
            # Destroying a node while callbacks are still unwinding triggers the
            # repeated 'cannot use Destroyable' errors seen in the previous run.
            print(
                "[camera-cleanup] executor thread still alive; skip node destruction",
                flush=True,
            )
        else:
            if self._camera_executor is not None and self._camera_node is not None:
                try:
                    self._camera_executor.remove_node(self._camera_node)
                except Exception:
                    pass
            if self._camera_node is not None:
                try:
                    self._camera_node.destroy_node()
                except Exception as exc:
                    print(f"[camera-cleanup] node destroy warning: {exc!r}", flush=True)

        self._camera_node = None
        self._camera_executor = None
        self._camera_thread = None


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

    def sample_next(
        self,
        dt: float,
        timeout_s: float,
        stop_event: threading.Event,
    ) -> SensorSnapshot | None:
        del timeout_s
        remaining = self._next - time.monotonic()
        if remaining > 0 and stop_event.wait(remaining):
            return None
        idx = self._index
        self._index += 1
        self._next += max(0.0, dt)
        height = int(self.config.dataset["image_height"])
        width = int(self.config.dataset["image_width"])
        full = np.zeros(36, dtype=np.float32)
        state = np.zeros(26, dtype=np.float32)
        images = {
            name: np.full((height, width, 3), (idx * 13) % 255, dtype=np.uint8)
            for name in self.config.camera_topics
        }
        return SensorSnapshot(
            sim_time=idx * max(0.0, dt),
            monotonic_time=time.monotonic(),
            state=state,
            full_state=full,
            images=images,
            joint_ages={},
            image_stamps={name: idx * max(0.0, dt) for name in images},
            camera_reused={name: False for name in images},
        )

    def close(self) -> None:
        pass
