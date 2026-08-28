"""Rabo VLA runtime."""

from .controller import VLAController


def run() -> None:
    VLAController().run()


__all__ = ["VLAController", "run"]
