"""Train and evaluate the lightweight Rabo behavior-cloning classifier."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .dataset import (
    CAMERA_NAMES,
    FULL_STATE_DIM,
    IMAGE_SIZE,
    STATE_DIM,
    BCObservationDataset,
    EpisodeData,
    load_episode,
    state_statistics,
)
from .model import BCClassifier, BCModelConfig, sequence_guard

MODEL_NAME = "RaboBC-VLA-v1"
ACTION_SPACE = "rabo_vla_action_v1"


def _parse_args() -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episode",
        type=Path,
        default=home / "vla_bridge/data/bc/episode_000000",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=home / "vla_bridge/models/rabo_bc_v1",
    )
    parser.add_argument("--expected-records", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--augmentations-per-record", type=int, default=100)
    parser.add_argument("--val-augmentations", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--proprio-noise-std", type=float, default=0.003)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--use-previous-action", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "epochs",
        "augmentations_per_record",
        "val_augmentations",
        "batch_size",
        "eval_every",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.samples_per_epoch < 0 or args.expected_records < 0 or args.num_workers < 0:
        raise ValueError("sample/record/worker counts cannot be negative")
    if args.lr <= 0 or args.weight_decay < 0 or args.proprio_noise_std < 0:
        raise ValueError("optimizer and noise values are invalid")
    if args.gradient_clip <= 0:
        raise ValueError("--gradient-clip must be positive")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _loader(
    dataset: BCObservationDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def _forward(
    model: BCClassifier,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["images"].to(device, non_blocking=True)
    state = batch["state"].to(device, non_blocking=True)
    target = batch["action_id"].to(device, non_blocking=True)
    previous = None
    if model.config.use_previous_action:
        previous = batch["previous_action_id"].to(device, non_blocking=True)
    return model(images, state, previous), target


def _train_epoch(
    model: BCClassifier,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    gradient_clip: float,
) -> dict[str, float]:
    model.train()
    total = 0
    correct = 0
    loss_sum = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits, target = _forward(model, batch, device)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        count = target.numel()
        total += count
        correct += int((logits.argmax(dim=1) == target).sum().item())
        loss_sum += float(loss.item()) * count
    return {"loss": loss_sum / total, "accuracy": correct / total}


@torch.inference_mode()
def _evaluate_loader(
    model: BCClassifier,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    confidence_sum = 0.0
    matrix = torch.zeros(
        (model.config.num_actions, model.config.num_actions), dtype=torch.int64
    )
    for batch in loader:
        logits, target = _forward(model, batch, device)
        loss = criterion(logits, target)
        probabilities = logits.softmax(dim=1)
        confidence, predicted = probabilities.max(dim=1)
        count = target.numel()
        total += count
        correct += int((predicted == target).sum().item())
        confidence_sum += float(confidence.sum().item())
        loss_sum += float(loss.item()) * count
        flat = (target * model.config.num_actions + predicted).detach().cpu()
        matrix += torch.bincount(
            flat, minlength=model.config.num_actions**2
        ).reshape(model.config.num_actions, model.config.num_actions)
    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "average_confidence": confidence_sum / total,
        "confusion_matrix": matrix.tolist(),
    }


@torch.inference_mode()
def _evaluate_original(
    model: BCClassifier,
    episode: EpisodeData,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    dataset = BCObservationDataset(episode, augment=False)
    loader = _loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=device,
    )
    model.eval()
    probabilities_by_id: list[torch.Tensor | None] = [None] * episode.num_actions
    for batch in loader:
        logits, _ = _forward(model, batch, device)
        probabilities = logits.softmax(dim=1).detach().cpu()
        for row, record_index in zip(
            probabilities, batch["record_index"].tolist(), strict=True
        ):
            probabilities_by_id[int(record_index)] = row

    if any(value is None for value in probabilities_by_id):
        raise RuntimeError("original evaluation did not produce every record")

    per_step: list[dict[str, Any]] = []
    matrix = torch.zeros((episode.num_actions, episode.num_actions), dtype=torch.int64)
    exact = 0
    phase_correct = 0
    confidence_sum = 0.0
    guarded_ids: list[int] = []
    cursor = 0
    for correct_id, (record, value) in enumerate(
        zip(episode.records, probabilities_by_id, strict=True)
    ):
        assert value is not None
        confidence, predicted = torch.max(value, dim=0)
        predicted_id = int(predicted.item())
        confidence_value = float(confidence.item())
        guarded_id, candidates = sequence_guard(value, cursor, max_advance=2)
        cursor = guarded_id
        guarded_ids.append(guarded_id)
        predicted_phase = episode.records[predicted_id].phase
        match = predicted_id == correct_id
        exact += int(match)
        phase_correct += int(predicted_phase == record.phase)
        confidence_sum += confidence_value
        matrix[correct_id, predicted_id] += 1
        per_step.append(
            {
                "step": correct_id,
                "phase": record.phase,
                "predicted_action_id": predicted_id,
                "correct_action_id": correct_id,
                "confidence": confidence_value,
                "match": match,
                "guarded_action_id": guarded_id,
                "guard_candidates": list(candidates),
            }
        )

    correct_sequence = list(range(episode.num_actions))
    guarded_matches = sum(
        int(predicted == correct)
        for predicted, correct in zip(guarded_ids, correct_sequence, strict=True)
    )
    return {
        "exact_accuracy": exact / episode.num_actions,
        "phase_accuracy": phase_correct / episode.num_actions,
        "average_confidence": confidence_sum / episode.num_actions,
        "confusion_matrix": matrix.tolist(),
        "per_step_prediction": per_step,
        "guarded_sequence": guarded_ids,
        "sequence_completion_accuracy": guarded_matches / episode.num_actions,
        "sequence_completion": guarded_ids == correct_sequence,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    _seed_everything(args.seed)
    expected = args.expected_records or None
    episode = load_episode(args.episode, expected_records=expected)
    output = args.output.expanduser().resolve()
    if (output / "model.pt").exists() and not args.overwrite:
        raise FileExistsError(f"model already exists; pass --overwrite to replace: {output}")
    output.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    samples_per_epoch = args.samples_per_epoch or (
        episode.num_actions * args.augmentations_per_record
    )
    model_config = BCModelConfig(
        num_actions=episode.num_actions,
        use_previous_action=bool(args.use_previous_action),
    )
    model = BCClassifier(model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_dataset = BCObservationDataset(
        episode,
        samples_per_epoch=samples_per_epoch,
        augment=True,
        proprio_noise_std=args.proprio_noise_std,
    )
    val_dataset = BCObservationDataset(
        episode,
        samples_per_epoch=episode.num_actions * args.val_augmentations,
        augment=True,
        proprio_noise_std=args.proprio_noise_std,
    )
    train_loader = _loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )
    val_loader = _loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )

    print(
        f"dataset records={episode.num_actions} images={episode.num_actions * len(CAMERA_NAMES)} "
        f"samples_per_epoch={samples_per_epoch}",
        flush=True,
    )
    print(
        f"model={MODEL_NAME} parameters={model.parameter_count} device={device} "
        f"uses_request_step_as_input=false",
        flush=True,
    )

    history: list[dict[str, Any]] = []
    best_score = (-1.0, -1.0)
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_train_accuracy = 0.0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = _train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.gradient_clip,
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
        }
        should_evaluate = epoch == 1 or epoch == args.epochs or epoch % args.eval_every == 0
        if should_evaluate:
            val_metrics = _evaluate_loader(model, val_loader, criterion, device)
            row.update(
                {
                    "augmented_val_loss": val_metrics["loss"],
                    "augmented_val_accuracy": val_metrics["accuracy"],
                    "augmented_val_confidence": val_metrics["average_confidence"],
                }
            )
            score = (val_metrics["accuracy"], train_metrics["accuracy"])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_train_accuracy = train_metrics["accuracy"]
                best_state = copy.deepcopy(
                    {name: value.detach().cpu() for name, value in model.state_dict().items()}
                )
            print(
                f"epoch={epoch:03d}/{args.epochs} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"aug_val_acc={val_metrics['accuracy']:.4f}",
                flush=True,
            )
        elif epoch % max(1, args.eval_every // 2) == 0:
            print(
                f"epoch={epoch:03d}/{args.epochs} "
                f"train_loss={train_metrics['loss']:.6f} "
                f"train_acc={train_metrics['accuracy']:.4f}",
                flush=True,
            )
        history.append(row)

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    trained_distribution = _evaluate_loader(model, train_loader, criterion, device)
    augmented = _evaluate_loader(model, val_loader, criterion, device)
    original = _evaluate_original(
        model,
        episode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    recommended = bool(
        original["exact_accuracy"] >= 0.95
        and augmented["accuracy"] >= 0.90
        and original["sequence_completion"]
    )

    stats = state_statistics(episode)
    config = {
        "model": MODEL_NAME,
        "model_family": "vision_language_action",
        "architecture": "shared_small_cnn_state_mlp_classifier",
        "num_actions": episode.num_actions,
        "cameras": list(CAMERA_NAMES),
        "image_size": IMAGE_SIZE,
        "proprio_dim": STATE_DIM,
        "full_proprio_dim": FULL_STATE_DIM,
        "trained_from_episodes": 1,
        "uses_request_step_as_input": False,
        "uses_previous_action_id": model_config.use_previous_action,
        "language_input": "accepted_not_encoded_v1",
        "action_space": ACTION_SPACE,
        "state_normalization": stats,
        "model_config": model_config.to_dict(),
        "parameters": model.parameter_count,
    }
    action_library = {
        "format": "rabo_bc_action_library_v1",
        "protocol": "rabo_command_v1",
        "action_space": ACTION_SPACE,
        "num_actions": episode.num_actions,
        "actions": [
            {
                "action_id": record.action_id,
                "phase": record.phase,
                "command": record.command,
            }
            for record in episode.records
        ],
    }
    metrics = {
        "model": MODEL_NAME,
        "dataset_records": episode.num_actions,
        "parameters": model.parameter_count,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "train_accuracy": trained_distribution["accuracy"],
        "train_loss": trained_distribution["loss"],
        "checkpoint_selection_train_accuracy": best_train_accuracy,
        "augmented_val_accuracy": augmented["accuracy"],
        "augmented_val_loss": augmented["loss"],
        "augmented_val_average_confidence": augmented["average_confidence"],
        "augmented_confusion_matrix": augmented["confusion_matrix"],
        "original_episode_exact_accuracy": original["exact_accuracy"],
        "original_episode_phase_accuracy": original["phase_accuracy"],
        "original_episode_average_confidence": original["average_confidence"],
        "original_confusion_matrix": original["confusion_matrix"],
        "per_step_prediction": original["per_step_prediction"],
        "guarded_sequence": original["guarded_sequence"],
        "sequence_completion_accuracy": original["sequence_completion_accuracy"],
        "sequence_completion": original["sequence_completion"],
        "recommended_for_online_test": recommended,
        "elapsed_s": time.time() - started,
        "history": history,
    }

    checkpoint = {
        "format": "rabo_bc_checkpoint_v1",
        "model_name": MODEL_NAME,
        "model_config": model_config.to_dict(),
        "state_dict": {name: value.cpu() for name, value in model.state_dict().items()},
        "state_normalization": stats,
        "uses_request_step_as_input": False,
    }
    temporary = output / "model.pt.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(output / "model.pt")
    _write_json(output / "config.json", config)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "action_library.json", action_library)

    print("confusion_matrix=" + json.dumps(original["confusion_matrix"]), flush=True)
    for item in original["per_step_prediction"]:
        print(
            "per_step "
            f"step={item['step']} predicted_action_id={item['predicted_action_id']} "
            f"correct_action_id={item['correct_action_id']} "
            f"confidence={item['confidence']:.6f} match={str(item['match']).lower()}",
            flush=True,
        )
    summary = {
        "dataset_records": episode.num_actions,
        "model": MODEL_NAME,
        "parameters": model.parameter_count,
        "epochs": args.epochs,
        "train_accuracy": trained_distribution["accuracy"],
        "augmented_val_accuracy": augmented["accuracy"],
        "original_episode_exact_accuracy": original["exact_accuracy"],
        "phase_accuracy": original["phase_accuracy"],
        "average_confidence": original["average_confidence"],
        "sequence_completion": original["sequence_completion"],
        "model_path": str(output / "model.pt"),
        "recommended_for_online_test": recommended,
    }
    print("training_summary=" + json.dumps(summary, ensure_ascii=False), flush=True)
    return metrics


def main() -> None:
    train(_parse_args())


if __name__ == "__main__":
    main()
