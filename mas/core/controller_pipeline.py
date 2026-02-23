"""Shared controller pipeline contract used across core and role layers."""

from enum import Enum, auto


class ControllerPipelineStep(Enum):
    """Controller internal state-machine steps."""

    IDLE = auto()
    ASSESS_NEEDS = auto()
    WAIT_FOR_WORKERS = auto()
    PLAN = auto()
    PUSH = auto()
    DONE = auto()
