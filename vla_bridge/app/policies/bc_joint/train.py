"""Train a small single-demonstration behavior-cloned joint policy.

This is intentionally a fixed-scene baseline. It learns a real neural mapping
from three RGB views + current 36D proprioception to the next 14D A7 arm joint
position target. O6 hand joints and request.step are never model inputs/targets.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import BCJointModel


CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


@dataclass
class Metrics:
    epochs: int
    train_loss: float
    train_mae_rad: float
    val_mae_rad: float
    original_episode_mae_rad: float
    original_episode_max_abs_rad: float
    progress_mae: float
    parameters: int
    device: str


class EpisodeDataset(Dataset):
    def __init__(
        self,
        cache: dict[str, np.ndarray],
        indices: np.ndarray,
        *,
        samples_per_epoch: int,
        augment: bool,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        target_mean: np.ndarray,
        target_std: np.ndarray,
    ) -> None:
        self.cache = cache
        self.indices = np.asarray(indices, dtype=np.int64)
        self.samples_per_epoch = max(len(self.indices), int(samples_per_epoch))
        self.augment = augment
        self.state_mean = torch.from_numpy(state_mean.astype(np.float32))
        self.state_std = torch.from_numpy(state_std.astype(np.float32))
        self.target_mean = torch.from_numpy(target_mean.astype(np.float32))
        self.target_std = torch.from_numpy(target_std.astype(np.float32))

    def __len__(self) -> int:
        return self.samples_per_epoch

    @staticmethod
    def _image(value: np.ndarray, augment: bool) -> torch.Tensor:
        # uint8 HWC RGB -> float CHW [0,1]
        x = torch.from_numpy(value.copy()).permute(2, 0, 1).float().div_(255.0)
        if augment:
            # Spatial semantics are preserved: no flips and no large rotations.
            _, h, w = x.shape
            crop_scale = random.uniform(0.92, 1.0)
            crop_h = max(8, int(h * crop_scale))
            crop_w = max(8, int(w * crop_scale))
            top = random.randint(0, max(0, h - crop_h))
            left = random.randint(0, max(0, w - crop_w))
            x = x[:, top : top + crop_h, left : left + crop_w]
            x = torch.nn.functional.interpolate(
                x.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(0)
            brightness = random.uniform(0.90, 1.10)
            contrast = random.uniform(0.90, 1.10)
            mean = x.mean(dim=(1, 2), keepdim=True)
            x = (x - mean) * contrast + mean
            x = x * brightness
            if random.random() < 0.5:
                x = x + torch.randn_like(x) * random.uniform(0.0, 0.01)
            x = x.clamp_(0.0, 1.0)
        return x

    def __getitem__(self, item: int):
        if self.augment:
            index = int(self.indices[random.randrange(len(self.indices))])
        else:
            index = int(self.indices[item % len(self.indices)])

        full = torch.from_numpy(self.cache["full_state"][index].astype(np.float32))
        target = torch.from_numpy(self.cache["target_arm_state"][index].astype(np.float32))
        if self.augment:
            full = full + torch.randn_like(full) * random.uniform(0.0, 0.003)

        normalized_state = (full - self.state_mean) / self.state_std
        normalized_target = (target - self.target_mean) / self.target_std
        images = [self._image(self.cache[key][index], self.augment) for key in CAMERA_KEYS]
        progress = torch.tensor(float(self.cache["progress"][index]), dtype=torch.float32)
        return (*images, normalized_state, normalized_target, progress)


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=False)
    required = {"full_state", "target_arm_state", "progress", *CAMERA_KEYS}
    missing = required.difference(raw.files)
    if missing:
        raise ValueError(f"BC cache missing arrays: {sorted(missing)}")
    cache = {key: np.asarray(raw[key]) for key in required}
    n = len(cache["full_state"])
    if cache["full_state"].shape != (n, 36):
        raise ValueError(f"full_state must be [N,36], got {cache['full_state'].shape}")
    if cache["target_arm_state"].shape != (n, 14):
        raise ValueError("target_arm_state must be [N,14]")
    for key in CAMERA_KEYS:
        if len(cache[key]) != n or cache[key].ndim != 4 or cache[key].shape[-1] != 3:
            raise ValueError(f"{key} must be [N,H,W,3], got {cache[key].shape}")
    return cache


def _forward_batch(model: BCJointModel, batch, device: torch.device):
    high, left, right, state, target, progress = batch
    inputs = [v.to(device, non_blocking=False) for v in (high, left, right, state)]
    target = target.to(device)
    progress = progress.to(device)
    action_pred, progress_pred = model(*inputs)
    return action_pred, progress_pred, target, progress


def _evaluate(
    model: BCJointModel,
    loader: DataLoader,
    device: torch.device,
    target_std: torch.Tensor,
) -> tuple[float, float]:
    model.eval()
    abs_sum = 0.0
    values = 0
    progress_sum = 0.0
    samples = 0
    with torch.no_grad():
        for batch in loader:
            pred, p_pred, target, progress = _forward_batch(model, batch, device)
            abs_rad = (pred - target).abs() * target_std.to(device)
            abs_sum += float(abs_rad.sum().item())
            values += int(abs_rad.numel())
            progress_sum += float((p_pred - progress).abs().sum().item())
            samples += int(progress.numel())
    return abs_sum / max(values, 1), progress_sum / max(samples, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path.home() / "vla_bridge" / "data" / "joint" / "bc_episode_v1.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "vla_bridge" / "models" / "rabo_bc_joint_v1",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        device = torch.device("cuda")
    elif args.device == "auto" and torch.cuda.is_available():
        try:
            # The target machine has previously shown a CUDA/driver mismatch;
            # perform a real allocation check and fall back to CPU if it fails.
            torch.empty(1, device="cuda")
            device = torch.device("cuda")
        except Exception:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    cache = _load_cache(args.cache.expanduser().resolve())
    n = len(cache["full_state"])
    # Deterministic interleaved validation frames preserve coverage of the whole
    # demonstration while leaving the neighboring frames for training.
    val_mask = np.arange(n) % 10 == 0
    train_indices = np.flatnonzero(~val_mask)
    val_indices = np.flatnonzero(val_mask)

    state_mean = cache["full_state"][train_indices].mean(axis=0).astype(np.float32)
    state_std = cache["full_state"][train_indices].std(axis=0).astype(np.float32)
    target_mean = cache["target_arm_state"][train_indices].mean(axis=0).astype(np.float32)
    target_std = cache["target_arm_state"][train_indices].std(axis=0).astype(np.float32)
    state_std = np.maximum(state_std, 0.02).astype(np.float32)
    target_std = np.maximum(target_std, 0.02).astype(np.float32)

    train_ds = EpisodeDataset(
        cache,
        train_indices,
        samples_per_epoch=args.samples_per_epoch,
        augment=True,
        state_mean=state_mean,
        state_std=state_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    val_ds = EpisodeDataset(
        cache,
        val_indices,
        samples_per_epoch=len(val_indices),
        augment=False,
        state_mean=state_mean,
        state_std=state_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    all_ds = EpisodeDataset(
        cache,
        np.arange(n),
        samples_per_epoch=n,
        augment=False,
        state_mean=state_mean,
        state_std=state_std,
        target_mean=target_mean,
        target_std=target_std,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    all_loader = DataLoader(all_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BCJointModel(proprio_dim=36, action_dim=14).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    action_loss_fn = nn.SmoothL1Loss(beta=0.5)
    progress_loss_fn = nn.MSELoss()
    target_std_tensor = torch.from_numpy(target_std)

    last_loss = math.nan
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for batch in train_loader:
            pred, p_pred, target, progress = _forward_batch(model, batch, device)
            loss_action = action_loss_fn(pred, target)
            loss_progress = progress_loss_fn(p_pred, progress)
            loss = loss_action + 0.10 * loss_progress
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item())
            batches += 1
        last_loss = running / max(batches, 1)

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            val_mae, val_progress = _evaluate(model, val_loader, device, target_std_tensor)
            print(
                f"epoch={epoch:03d} loss={last_loss:.6f} "
                f"val_mae_rad={val_mae:.6f} val_progress_mae={val_progress:.5f}",
                flush=True,
            )

    train_eval_ds = EpisodeDataset(
        cache,
        train_indices,
        samples_per_epoch=len(train_indices),
        augment=False,
        state_mean=state_mean,
        state_std=state_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, num_workers=0)
    train_mae, _ = _evaluate(model, train_eval_loader, device, target_std_tensor)
    val_mae, progress_mae = _evaluate(model, val_loader, device, target_std_tensor)
    original_mae, _ = _evaluate(model, all_loader, device, target_std_tensor)

    # Also compute the maximum original-episode absolute error in radians.
    model.eval()
    max_abs = 0.0
    with torch.no_grad():
        for batch in all_loader:
            pred, _, target, _ = _forward_batch(model, batch, device)
            error = (pred - target).abs() * target_std_tensor.to(device)
            max_abs = max(max_abs, float(error.max().item()))

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "state_mean": torch.from_numpy(state_mean),
        "state_std": torch.from_numpy(state_std),
        "target_mean": torch.from_numpy(target_mean),
        "target_std": torch.from_numpy(target_std),
        "image_size": list(cache[CAMERA_KEYS[0]].shape[1:3]),
        "action_dim": 14,
        "proprio_dim": 36,
        "camera_keys": list(CAMERA_KEYS),
    }
    torch.save(checkpoint, output / "model.pt")

    parameters = sum(p.numel() for p in model.parameters())
    metrics = Metrics(
        epochs=args.epochs,
        train_loss=float(last_loss),
        train_mae_rad=float(train_mae),
        val_mae_rad=float(val_mae),
        original_episode_mae_rad=float(original_mae),
        original_episode_max_abs_rad=float(max_abs),
        progress_mae=float(progress_mae),
        parameters=int(parameters),
        device=str(device),
    )
    (output / "metrics.json").write_text(
        json.dumps(asdict(metrics), indent=2), encoding="utf-8"
    )
    config = {
        "model": "RaboBC-Joint-v1",
        "model_family": "behavior_cloning",
        "action_space": "arm_joint_position_14d",
        "action_dim": 14,
        "proprio_dim": 36,
        "vision_inputs": 3,
        "camera_keys": list(CAMERA_KEYS),
        "trained_from_episodes": 1,
        "training_frames": n,
        "uses_request_step_as_input": False,
        "action_definition": "next left_arm(7) + right_arm(7) joint position",
        "controls_o6_hand_joints": False,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps({**config, **asdict(metrics)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
