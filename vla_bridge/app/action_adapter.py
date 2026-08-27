from __future__ import annotations

import math
from collections.abc import Sequence


class RobotActionAdapter:
    """Maps a model-native action chunk to the bridge's immediate action.

    The real Linker A7 + O6 mapping deliberately does not live here yet.  Raw
    mode returns the first model-native action.  compat7 is an explicitly
    temporary transport compatibility mode that crops or zero-pads to 7D.
    """

    def __init__(self, output_mode: str = "raw") -> None:
        if output_mode not in {"raw", "compat7"}:
            raise ValueError(f"Unsupported VLA_OUTPUT_MODE: {output_mode}")
        self.output_mode = output_mode

    def adapt(self, action_chunk: Sequence[Sequence[float]]) -> list[float]:
        if not action_chunk:
            raise ValueError("Model returned an empty action chunk")

        rows = [[float(value) for value in row] for row in action_chunk]
        width = len(rows[0])
        if width == 0 or any(len(row) != width for row in rows):
            raise ValueError("Model returned an empty or ragged action chunk")
        if any(not math.isfinite(value) for row in rows for value in row):
            raise ValueError("Model returned NaN or Inf action values")

        action = rows[0]
        if self.output_mode == "compat7":
            return (action[:7] + [0.0] * 7)[:7]
        return action

    def output_dim(self, native_dim: int) -> int:
        return 7 if self.output_mode == "compat7" else native_dim
