"""Backward-compatible import for the original transport diagnostic policy."""

from .policies.zero_policy import ZeroPolicy

__all__ = ["ZeroPolicy"]
