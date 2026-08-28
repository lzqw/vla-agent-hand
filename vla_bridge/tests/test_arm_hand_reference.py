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


def test_alignment_is_observation_driven_and_events_fire_once(tmp_path):
    reference, reference_path, event_path = _artifacts(tmp_path)
    trajectory = ArmHandReference(
        reference_path, event_path, initial_search=6, forward_window=4
    )

    first = trajectory.align("episode-a", reference[0])
    assert first.reference_index == 0
    assert first.target_arm[0] == 1.0
    assert first.hand_command["action_type"] == "right_hand_clench"

    # Repeated shadow observations must not advance the server-side cursor.
    repeated = trajectory.align("episode-a", reference[0])
    assert repeated.reference_index == 0
    assert repeated.hand_command is None

    crossed = trajectory.align("episode-a", reference[3])
    assert crossed.reference_index == 3
    assert crossed.hand_command["action_type"] == "right_hand_grasp_force"
    assert trajectory.align("episode-a", reference[3]).hand_command is None

    trajectory.reset("episode-a")
    assert trajectory.align("episode-a", reference[0]).hand_command is not None


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
