"""Dataset validation and stochastic augmentation for one Rabo Expert episode."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

CAMERA_NAMES = ("cam_high", "cam_left_wrist", "cam_right_wrist")
STATE_DIM = 26
FULL_STATE_DIM = 36
IMAGE_SIZE = 224
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class BCRecord:
    action_id: int
    phase: str
    instruction: str
    state: tuple[float, ...]
    full_state: tuple[float, ...]
    image_paths: tuple[Path, ...]
    command: dict[str, Any]


@dataclass(frozen=True)
class EpisodeData:
    root: Path
    meta: dict[str, Any]
    expert_program: dict[str, Any]
    records: tuple[BCRecord, ...]

    @property
    def num_actions(self) -> int:
        return len(self.records)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required dataset file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _vector(value: Any, *, name: str, size: int) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, dict)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a JSON array")
    if len(value) != size:
        raise ValueError(f"{name} must contain {size} values, got {len(value)}")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains a non-numeric value") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def _resolve_image(root: Path, raw_path: Any, *, name: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{name} must be a non-empty relative path")
    path = (root / raw_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} escapes episode directory: {raw_path}") from exc
    if not path.is_file():
        raise ValueError(f"{name} does not exist: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.load()
            if rgb.width <= 0 or rgb.height <= 0:
                raise ValueError(f"{name} has an invalid image size")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"{name} is not a readable image: {path}") from exc
    return path


def load_episode(episode_dir: Path, expected_records: int | None = None) -> EpisodeData:
    """Validate and load a successful command-level Expert episode.

    The record ``step`` is used only as the supervised class label and for
    evaluation.  It is never returned as a model input feature.
    """

    root = episode_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"episode directory does not exist: {root}")

    meta = _read_json(root / "meta.json")
    expert = _read_json(root / "expert_program.json")
    if expert.get("format") != "rabo_expert_program_v1":
        raise ValueError("expert_program.json has an unsupported format")
    if expert.get("protocol") != "rabo_command_v1":
        raise ValueError("expert_program.json has an unsupported protocol")
    commands = expert.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError("expert_program.json commands must be a non-empty array")
    if expert.get("num_steps") != len(commands):
        raise ValueError("expert_program.json num_steps must equal len(commands)")

    steps_path = root / "steps.jsonl"
    try:
        lines = steps_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"required dataset file is missing: {steps_path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"steps.jsonl line {line_number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise TypeError(f"steps.jsonl line {line_number} must be an object")
        rows.append(row)

    if not rows:
        raise ValueError("steps.jsonl contains no records")
    if expected_records is not None and len(rows) != expected_records:
        raise ValueError(f"expected {expected_records} records, found {len(rows)}")
    if len(rows) != len(commands):
        raise ValueError(
            f"steps records ({len(rows)}) do not match Expert commands ({len(commands)})"
        )
    if meta.get("num_steps") not in {None, len(rows)}:
        raise ValueError("meta.json num_steps does not match steps.jsonl")

    records: list[BCRecord] = []
    for action_id, (row, item) in enumerate(zip(rows, commands, strict=True)):
        if row.get("step") != action_id:
            raise ValueError(f"record {action_id} has non-sequential step={row.get('step')!r}")
        if not isinstance(item, dict) or item.get("step") != action_id:
            raise ValueError(f"Expert command {action_id} has a non-sequential step")
        if row.get("execution_success") is not True:
            raise ValueError(f"record {action_id} execution_success is not true")

        phase = row.get("phase")
        instruction = row.get("instruction")
        observation = row.get("observation")
        if not isinstance(phase, str) or not phase:
            raise ValueError(f"record {action_id} phase must be non-empty")
        if phase != item.get("phase"):
            raise ValueError(f"record {action_id} phase differs from expert_program.json")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"record {action_id} instruction must be non-empty")
        if not isinstance(observation, dict):
            raise TypeError(f"record {action_id} observation must be an object")

        state = _vector(
            observation.get("state"), name=f"record {action_id} state", size=STATE_DIM
        )
        full_state = _vector(
            observation.get("full_state"),
            name=f"record {action_id} full_state",
            size=FULL_STATE_DIM,
        )
        images = observation.get("images")
        if not isinstance(images, dict):
            raise TypeError(f"record {action_id} images must be an object")
        image_paths = tuple(
            _resolve_image(
                root,
                images.get(camera),
                name=f"record {action_id} images.{camera}",
            )
            for camera in CAMERA_NAMES
        )

        command = row.get("command")
        if not isinstance(command, dict) or not command.get("action_type"):
            raise ValueError(f"record {action_id} command must contain action_type")
        if command != item.get("command"):
            raise ValueError(f"record {action_id} command differs from expert_program.json")
        records.append(
            BCRecord(
                action_id=action_id,
                phase=phase,
                instruction=instruction,
                state=state,
                full_state=full_state,
                image_paths=image_paths,
                command=command,
            )
        )

    return EpisodeData(root=root, meta=meta, expert_program=expert, records=tuple(records))


class AddGaussianNoise:
    def __init__(self, std: float = 0.005) -> None:
        self.std = float(std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        return (tensor + torch.randn_like(tensor) * self.std).clamp_(0.0, 1.0)


def build_image_transform(*, augment: bool, image_size: int = IMAGE_SIZE):
    if augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.92, 1.0),
                    ratio=(0.98, 1.02),
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.ColorJitter(
                    brightness=0.10,
                    contrast=0.10,
                    saturation=0.05,
                    hue=0.0,
                ),
                transforms.RandomAffine(
                    degrees=0.0,
                    translate=(0.018, 0.018),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
                transforms.ToTensor(),
                AddGaussianNoise(0.005),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.load()
        return rgb


def state_statistics(episode: EpisodeData) -> dict[str, list[float]]:
    values = torch.tensor([record.state for record in episode.records], dtype=torch.float32)
    std = values.std(dim=0, unbiased=False).clamp_min_(1.0e-4)
    return {
        "mean": values.mean(dim=0).tolist(),
        "std": std.tolist(),
        "min": values.min(dim=0).values.tolist(),
        "max": values.max(dim=0).values.tolist(),
    }


class BCObservationDataset(Dataset[dict[str, torch.Tensor]]):
    """Dynamically samples and augments the original observations."""

    def __init__(
        self,
        episode: EpisodeData,
        *,
        samples_per_epoch: int | None = None,
        augment: bool,
        proprio_noise_std: float = 0.003,
        image_size: int = IMAGE_SIZE,
        cache_images: bool = True,
    ) -> None:
        self.episode = episode
        self.samples_per_epoch = samples_per_epoch or len(episode.records)
        if self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        self.random_sample = self.samples_per_epoch != len(episode.records)
        self.augment = bool(augment)
        self.proprio_noise_std = float(proprio_noise_std) if augment else 0.0
        self.transform = build_image_transform(augment=augment, image_size=image_size)

        stats = state_statistics(episode)
        self.state_mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.state_std = torch.tensor(stats["std"], dtype=torch.float32)
        self.state_min = torch.tensor(stats["min"], dtype=torch.float32)
        self.state_max = torch.tensor(stats["max"], dtype=torch.float32)
        self._cached_images: tuple[tuple[Image.Image, ...], ...] | None = None
        if cache_images:
            self._cached_images = tuple(
                tuple(_load_rgb(path) for path in record.image_paths)
                for record in episode.records
            )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _record_index(self, index: int) -> int:
        if self.random_sample:
            return int(torch.randint(len(self.episode.records), (1,)).item())
        return index % len(self.episode.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index = self._record_index(index)
        record = self.episode.records[record_index]
        if self._cached_images is None:
            images = tuple(_load_rgb(path) for path in record.image_paths)
        else:
            images = tuple(image.copy() for image in self._cached_images[record_index])
        image_tensor = torch.stack([self.transform(image) for image in images], dim=0)

        state = torch.tensor(record.state, dtype=torch.float32)
        if self.proprio_noise_std > 0:
            state = state + torch.randn_like(state) * self.proprio_noise_std
            state = torch.maximum(torch.minimum(state, self.state_max), self.state_min)
        normalized_state = (state - self.state_mean) / self.state_std
        previous_action_id = (
            record.action_id - 1 if record.action_id > 0 else self.episode.num_actions
        )
        return {
            "images": image_tensor,
            "state": normalized_state,
            "action_id": torch.tensor(record.action_id, dtype=torch.long),
            "previous_action_id": torch.tensor(previous_action_id, dtype=torch.long),
            "record_index": torch.tensor(record_index, dtype=torch.long),
        }
