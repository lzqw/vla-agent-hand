from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class JointSpec:
    name: str
    topic: str
    ros_name: str


@dataclass(frozen=True)
class JointGroup:
    side: str
    joints: tuple[JointSpec, ...]


@dataclass(frozen=True)
class CollectorConfig:
    raw: dict[str, Any]
    source_path: Path

    @property
    def dataset(self) -> dict[str, Any]:
        return self.raw["dataset"]

    @property
    def rabo(self) -> dict[str, Any]:
        return self.raw["rabo"]

    @property
    def collection(self) -> dict[str, Any]:
        return self.raw["collection"]

    @property
    def fps(self) -> int:
        return int(self.dataset["fps"])

    @property
    def root(self) -> Path:
        value = Path(self.dataset["root"])
        if not value.is_absolute():
            value = (self.source_path.parent / value).resolve()
        return value

    @property
    def active_hand_indices(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.raw["state"]["active_hand_indices"])

    @property
    def hand_joint_names(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.raw["state"]["hand_joint_names"])

    @property
    def joint_groups(self) -> tuple[JointGroup, ...]:
        groups: list[JointGroup] = []
        for side, items in self.raw["state"]["joints"].items():
            groups.append(
                JointGroup(
                    str(side),
                    tuple(
                        JointSpec(str(j["name"]), str(j["topic"]), str(j["ros_name"]))
                        for j in items
                    ),
                )
            )
        return tuple(groups)

    @property
    def full_state_names(self) -> tuple[str, ...]:
        return tuple(j.name for group in self.joint_groups for j in group.joints)

    @property
    def state_names(self) -> tuple[str, ...]:
        active = self.active_hand_indices
        names: list[str] = []
        for group in self.joint_groups:
            if group.side in ("left_arm", "right_arm"):
                names.extend(j.name for j in group.joints)
            else:
                names.extend(group.joints[index].name for index in active)
        return tuple(names)

    @property
    def camera_topics(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in self.dataset["cameras"].items()}

    @property
    def primary_camera(self) -> str:
        value = self.dataset.get("primary_camera")
        if value is None:
            return next(iter(self.camera_topics))
        return str(value)

    @property
    def sample_wait_timeout_s(self) -> float:
        return float(self.dataset.get("sample_wait_timeout_s", 2.0))

    @property
    def ring_buffer_depth(self) -> int:
        return int(self.dataset.get("ring_buffer_depth", 16))

    @property
    def writer_queue_size(self) -> int:
        return int(self.dataset.get("writer_queue_size", 32))

    @property
    def max_joint_age_s(self) -> float:
        return float(self.dataset.get("max_joint_age_s", 0.2))

    @property
    def max_camera_skew_s(self) -> float:
        return float(self.dataset.get("max_camera_skew_s", 0.2))

    @property
    def max_stamp_gap_s(self) -> float:
        return float(self.dataset.get("max_stamp_gap_s", 0.2))

    def validate(self) -> None:
        if self.fps <= 0:
            raise ValueError("dataset.fps 必须大于0")
        if self.active_hand_indices != (0, 1, 3, 5, 7, 9):
            raise ValueError("O6主动关节索引必须为 [0,1,3,5,7,9]")
        if len(self.hand_joint_names) != 11:
            raise ValueError("state.hand_joint_names 必须包含O6全部11个关节")
        sides = tuple(group.side for group in self.joint_groups)
        if sides != ("left_arm", "right_arm", "left_hand", "right_hand"):
            raise ValueError("state.joints 必须按 left_arm/right_arm/left_hand/right_hand 顺序")
        counts = tuple(len(group.joints) for group in self.joint_groups)
        if counts != (7, 7, 11, 11):
            raise ValueError(f"state.joints 关节数必须为 7/7/11/11，实际 {counts}")
        if len(set(self.full_state_names)) != len(self.full_state_names):
            raise ValueError("state.joints 中存在重复关节语义名")
        topics = [j.topic for group in self.joint_groups for j in group.joints]
        if len(set(topics)) != len(topics):
            raise ValueError("state.joints 中存在重复 topic")
        if len(self.state_names) != 26 or len(self.full_state_names) != 36:
            raise ValueError("state/full_state 维度必须分别为26/36")
        if not self.camera_topics:
            raise ValueError("至少需要配置一个相机话题")
        if self.primary_camera not in self.camera_topics:
            raise ValueError(f"primary_camera={self.primary_camera!r} 不在 cameras 中")
        if int(self.dataset["image_width"]) <= 0 or int(self.dataset["image_height"]) <= 0:
            raise ValueError("图像尺寸必须大于0")


def load_config(path: str | Path) -> CollectorConfig:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = CollectorConfig(raw=raw, source_path=source)
    config.validate()
    return config
