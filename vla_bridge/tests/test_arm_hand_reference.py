from __future__ import annotations

import json

import numpy as np

from app.policies.arm_hand_reference import ArmHandReference
from tools.prepare_joint_dataset import _phase_constrained_alignment


def _artifacts(tmp_path):
    reference = np.zeros((6, 14), dtype=np.float32)
    reference[:, 0] = np.arange(6, dtype=np.float32)
    target = np.concatenate([reference[1:], reference[-1:]], axis=0)
    reference_path = tmp_path / "arm_hand_reference_v1.npz"
    np.savez_compressed(
        reference_path,
        reference_arm_state=reference,
        target_arm_state=target,
    )
    event_path = tmp_path / "hand_events_v1.json"
    event_path.write_text(
        json.dumps(
            {
                "format": "rabo_hand_events_v1",
                "reference_frames": 6,
                "events": [
                    {
                        "event_id": 0,
                        "frame_index": 0,
                        "expert_step": 1,
                        "phase": "grasp",
                        "command": {"action_type": "right_hand_clench", "clench": [1.0]},
                    },
                    {
                        "event_id": 1,
                        "frame_index": 2,
                        "expert_step": 2,
                        "phase": "grasp",
                        "command": {
                            "action_type": "right_hand_grasp_force",
                            "strength": 0.4,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return reference, reference_path, event_path


def _legacy_fast(reference_path, event_path, **kwargs):
    return ArmHandReference(
        reference_path,
        event_path,
        hand_event_settle_cycles=1,
        grasp_force_repeats=1,
        **kwargs,
    )


def test_alignment_is_observation_driven_and_events_fire_once(tmp_path):
    reference, reference_path, event_path = _artifacts(tmp_path)
    trajectory = _legacy_fast(
        reference_path, event_path, initial_search=6, forward_window=4
    )

    first = trajectory.align("episode-a", reference[0])
    assert first.reference_index == 0
    assert first.action_reference_index == 0
    assert first.target_arm[0] == 0.0
    assert first.hand_command["action_type"] == "right_hand_clench"

    repeated = trajectory.align("episode-a", reference[0])
    assert repeated.reference_index == 0
    assert repeated.action_reference_index == 2
    assert repeated.hand_command is None

    crossed = trajectory.align("episode-a", reference[3])
    assert crossed.reference_index == 3
    assert crossed.hand_command["action_type"] == "right_hand_grasp_force"
    assert crossed.action_reference_index == 2
    assert trajectory.align("episode-a", reference[3]).hand_command is None

    trajectory.reset("episode-a")
    assert trajectory.align("episode-a", reference[0]).hand_command is not None


def test_lookahead_does_not_skip_pending_hand_event(tmp_path):
    reference, reference_path, event_path = _artifacts(tmp_path)
    trajectory = _legacy_fast(
        reference_path,
        event_path,
        initial_search=6,
        forward_window=4,
        lookahead_frames=3,
    )

    first = trajectory.align("episode-lookahead", reference[0])
    assert first.hand_event_id == 0
    assert first.action_reference_index == 0

    second = trajectory.align("episode-lookahead", reference[0])
    assert second.hand_command is None
    assert second.reference_index == 0
    assert second.action_reference_index == 2
    assert second.target_arm[0] == 2.0

    event = trajectory.align("episode-lookahead", reference[2])
    assert event.hand_event_id == 1
    assert event.action_reference_index == 2
    assert event.target_arm[0] == 2.0


def test_hand_event_waits_for_convergence_and_settle(tmp_path):
    reference, reference_path, event_path = _artifacts(tmp_path)
    trajectory = ArmHandReference(
        reference_path,
        event_path,
        initial_search=6,
        forward_window=4,
        lookahead_frames=3,
        hand_event_tolerance_rad=0.05,
        hand_event_settle_cycles=2,
        grasp_force_repeats=2,
    )

    # First event is due at frame 0, but one settle cycle is required before it fires.
    first = trajectory.align("episode-gated", reference[0])
    assert first.hand_command is None
    assert first.hand_event_id == 0
    assert first.hand_event_settle_count == 1
    assert first.action_reference_index == 0

    second = trajectory.align("episode-gated", reference[0])
    assert second.hand_command["action_type"] == "right_hand_clench"
    assert second.hand_event_settle_count == 2
    assert second.action_reference_index == 0

    # Move near frame 2 but outside event tolerance: event remains pending and arms hold frame 2.
    off = reference[2].copy()
    off[0] += 0.2
    pending = trajectory.align("episode-gated", off)
    assert pending.hand_command is None
    assert pending.hand_event_id == 1
    assert pending.hand_event_error_rad > 0.05
    assert pending.action_reference_index == 2

    # Two converged cycles settle the grasp event; then grasp_force is emitted twice.
    settle = trajectory.align("episode-gated", reference[2])
    assert settle.hand_command is None
    assert settle.hand_event_settle_count == 1

    grasp1 = trajectory.align("episode-gated", reference[2])
    assert grasp1.hand_command["action_type"] == "right_hand_grasp_force"
    assert grasp1.hand_event_repeat_index == 1
    assert grasp1.action_reference_index == 2

    grasp2 = trajectory.align("episode-gated", reference[2])
    assert grasp2.hand_command["action_type"] == "right_hand_grasp_force"
    assert grasp2.hand_event_repeat_index == 2
    assert grasp2.action_reference_index == 2

    after = trajectory.align("episode-gated", reference[2])
    assert after.hand_command is None
    assert after.action_reference_index > 2


def test_reached_target_advances_across_duplicate_reference_frames(tmp_path):
    reference, reference_path, event_path = _artifacts(tmp_path)
    reference[1] = reference[0]
    target = np.concatenate([reference[1:], reference[-1:]], axis=0)
    np.savez_compressed(
        reference_path,
        reference_arm_state=reference,
        target_arm_state=target,
    )
    trajectory = _legacy_fast(reference_path, event_path, initial_search=6)
    first = trajectory.align("episode-plateau", reference[0])
    second = trajectory.align("episode-plateau", reference[0])
    assert first.reference_index == 0
    assert second.reference_index == 1


def test_sparse_to_dense_mapping_is_phase_constrained_and_monotonic():
    reference = np.arange(12, dtype=np.float32)[:, None]
    query = reference[[0, 3, 7, 11]]
    reference_phases = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    query_phases = ["a", "a", "b", "c"]
    mapping, error, selected_phases = _phase_constrained_alignment(
        query, query_phases, reference, reference_phases
    )
    assert mapping.tolist() == [0, 3, 7, 11]
    assert np.all(error == 0.0)
    assert selected_phases == query_phases
