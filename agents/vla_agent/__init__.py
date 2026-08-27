"""Rabo -> 4080 fixed-oracle / future-VLA executable agent."""

from .controller import RemoteVLAController


def run() -> None:
    """Platform entry: connect 4080, then execute the remote closed loop."""
    RemoteVLAController().run()


__all__ = ["RemoteVLAController", "run"]
