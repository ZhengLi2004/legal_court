"""Shared controller pipeline contract used across core and role layers."""

from enum import Enum, auto


class ControllerPipelineStep(Enum):
    """Controller internal state-machine steps.

    Attributes:
        IDLE: No pending controller work.
        ASSESS_NEEDS: Assess worker and graph needs.
        WAIT_FOR_WORKERS: Wait for worker outputs.
        PLAN: Plan next controller actions.
        PUSH: Push actions into the graph.
        DONE: Finish the current controller cycle.
    """

    IDLE = auto()
    ASSESS_NEEDS = auto()
    WAIT_FOR_WORKERS = auto()
    PLAN = auto()
    PUSH = auto()
    DONE = auto()
