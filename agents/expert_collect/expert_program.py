"""Fixed-scene expert program matching the rabo_command_v1 remote protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FIXED_NUT_POSES: dict[str, tuple[float, float, float, float, float, float]] = {
    "B": (-0.3413, -0.1710, 0.2806, 0.0, 0.0, 0.5233),
    "A": (-0.2286, -0.0999, 0.2819, 0.0, 0.0, 0.5233),
    "C": (-0.2975, -0.0527, 0.2872, 0.0, 0.0, 0.5233),
}
RIGHT_ARM_BASE = (-0.6816, -0.004)

@dataclass(frozen=True)
class NutMotion:
    approach_z: float
    handoff_offset: float
    right_fingers: tuple[int, ...]
    right_strength: float
    fine_start_z: float
    fine_grasp_z: float
    left_strength: float
    right_retreat_z: float
    after_handoff: tuple[float, float, float, float, float, float] | None
    place_waypoints: tuple[tuple[float, float, float, float, float, float], ...]
    retreat: tuple[float, float, float, float, float, float]

MOTIONS: dict[str, NutMotion] = {
    "B": NutMotion(-0.331, 0.0, (1,2,3,4), 0.4, -0.2, -0.185, 1.0, 0.12,
        (0.44,0.06,-0.05,0.0,1.3,1.57), ((0.38,0.25,-0.22,0.0,-0.8,0.0),),
        (0.38,0.25,-0.12,0.0,-0.8,0.0)),
    "C": NutMotion(-0.33, 0.0062, (1,2,3,4), 0.8, -0.25, -0.185, 1.0, 0.0962,
        (0.44,0.06,-0.1,0.0,1.3,1.57),
        ((0.34,0.25,-0.12,0.0,0.0,0.0),(0.34,0.25,-0.2,0.0,0.0,0.0)),
        (0.34,0.25,-0.1,0.0,0.0,0.0)),
    "A": NutMotion(-0.3268, 0.0009, (1,2,3,4,5), 0.5, -0.2, -0.180, 1.0, 0.0509,
        None, ((0.43,0.25,-0.12,0.0,0.0,0.0),(0.43,0.25,-0.2,0.0,0.0,0.0)),
        (0.43,0.25,-0.1,0.0,0.0,0.0)),
}

def _move(label: str, pose: tuple[float, ...]) -> dict[str, Any]:
    return {"label": label, "pose": [float(v) for v in pose]}

def _item(phase: str, command: dict[str, Any]) -> dict[str, Any]:
    return {"phase": phase, "command": command}

def build_expert_program(nuts: tuple[str, ...] = ("B","C","A")) -> list[dict[str, Any]]:
    """Build the deterministic 59-step fixed-scene expert program.

    The 0.40 s handoff settle and 0.15 s post-grasp wait intentionally improve
    robustness against sticky simulated contact dynamics.
    """
    program: list[dict[str, Any]] = []
    for index, name in enumerate(nuts):
        motion = MOTIONS[name]
        position = FIXED_NUT_POSES[name]
        target_x = RIGHT_ARM_BASE[0] - position[0] + 0.06 - (0.02 if name == "A" else 0.0)
        target_y = RIGHT_ARM_BASE[1] - position[1] - 0.01
        offset = motion.handoff_offset
        program += [
            _item(f"approach_{name}", {"action_type":"right_arm_move_to","right_moves":[_move(f"右臂接近{name}",(target_x,target_y,motion.approach_z,0.0,0.8,0.0))]}),
            _item(f"right_grasp_{name}", {"action_type":"right_hand_clench","clench":[1.0,None,None,None,None,None]}),
            _item(f"right_grasp_{name}", {"action_type":"right_hand_grasp_force","strength":motion.right_strength,"fingers":list(motion.right_fingers)}),
            _item(f"lift_and_left_approach_{name}", {"action_type":"parallel_arm_sequence","right_moves":[_move(f"右臂{name}提起",(-0.41,0.12,-0.03+offset,0.0,0.8,0.0)),_move(f"右臂{name}进入交接位",(-0.41,-0.005,-0.03+offset,0.0,0.8,0.0))],"left_moves":[_move(f"左臂{name}粗接近",(0.43,0.3,-0.1+offset,0.0,1.3,1.57))]}),
            _item(f"left_fine_approach_{name}", {"action_type":"left_arm_move_to","left_moves":[_move(f"左臂{name}精接近",(0.44,0.06,motion.fine_start_z,0.0,1.3,1.57))]}),
            _item(f"left_fine_approach_{name}", {"action_type":"left_hand_clench","clench":[0.0,0.3,0.3,0.3,0.3,0.3]}),
            _item(f"left_fine_approach_{name}", {"action_type":"left_arm_move_to","left_moves":[_move(f"左臂{name}接取高度",(0.44,0.06,motion.fine_grasp_z,0.0,1.3,1.57))]}),
            _item(f"handoff_{name}_right_release", {"action_type":"right_hand_clench","clench":[0.9,0.0,0.0,0.0,0.0,0.0]}),
            _item(f"handoff_{name}_right_lift_clear", {"action_type":"right_arm_move_to","right_moves":[_move(f"右臂{name}释放后抬起",(-0.4,0.0,motion.right_retreat_z,0.0,0.8,0.0))]}),
            _item(f"handoff_{name}_settle", {"action_type":"wait","duration_s":0.40}),
            _item(f"handoff_{name}_left_grasp", {"action_type":"left_hand_clench","clench":[0.0,0.5,0.5,0.5,0.5,0.5]}),
            _item(f"handoff_{name}_left_grasp", {"action_type":"left_hand_grasp_force","strength":motion.left_strength}),
            _item(f"handoff_{name}_left_grasp", {"action_type":"wait","duration_s":0.15}),
        ]
        if motion.after_handoff is not None:
            program.append(_item(f"place_{name}", {"action_type":"left_arm_move_to","left_moves":[_move("左臂路径点",motion.after_handoff)]}))
        for waypoint in motion.place_waypoints:
            program.append(_item(f"place_{name}", {"action_type":"left_arm_move_to","left_moves":[_move("左臂路径点",waypoint)]}))
        program += [
            _item(f"release_{name}", {"action_type":"left_hand_clench","clench":[0.0,0.0,0.0,0.0,0.0,0.0]}),
            _item(f"release_{name}", {"action_type":"wait","duration_s":0.20}),
            _item(f"release_{name}", {"action_type":"left_arm_move_to","left_moves":[_move("左臂路径点",motion.retreat)]}),
        ]
        if index < len(nuts)-1:
            program.append(_item(f"wait_after_{name}", {"action_type":"wait","duration_s":1.0}))
    program.append(_item("completed", {"action_type":"wait","duration_s":0.15}))
    program.append(_item("completed", {"action_type":"done"}))
    return program
