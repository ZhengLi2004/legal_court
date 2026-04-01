"""Session status enum and transition constraints."""

from __future__ import annotations

from enum import Enum
from typing import Dict, Set


class SessionStatus(str, Enum):
    """Define canonical lifecycle statuses for one runtime session.

    Attributes:
        CREATED: Session created but not yet initialized.
        SETTING_UP: Setup is currently running.
        SETUP_DONE: Setup completed successfully.
        DEBATING: Debate loop is active.
        READY_FOR_ADJUDICATION: Ready to call the judge.
        FINISHED: Debate and adjudication finished.
        ERROR: Session entered an error state.
    """

    CREATED = "CREATED"
    SETTING_UP = "SETTING_UP"
    SETUP_DONE = "SETUP_DONE"
    DEBATING = "DEBATING"
    READY_FOR_ADJUDICATION = "READY_FOR_ADJUDICATION"
    FINISHED = "FINISHED"
    ERROR = "ERROR"


ALLOWED_TRANSITIONS: Dict[SessionStatus, Set[SessionStatus]] = {
    SessionStatus.CREATED: {SessionStatus.SETTING_UP, SessionStatus.ERROR},
    SessionStatus.SETTING_UP: {
        SessionStatus.SETUP_DONE,
        SessionStatus.DEBATING,
        SessionStatus.READY_FOR_ADJUDICATION,
        SessionStatus.FINISHED,
        SessionStatus.ERROR,
    },
    SessionStatus.SETUP_DONE: {
        SessionStatus.DEBATING,
        SessionStatus.READY_FOR_ADJUDICATION,
        SessionStatus.FINISHED,
        SessionStatus.ERROR,
    },
    SessionStatus.DEBATING: {
        SessionStatus.DEBATING,
        SessionStatus.READY_FOR_ADJUDICATION,
        SessionStatus.FINISHED,
        SessionStatus.ERROR,
    },
    SessionStatus.READY_FOR_ADJUDICATION: {
        SessionStatus.FINISHED,
        SessionStatus.ERROR,
    },
    SessionStatus.FINISHED: set(),
    SessionStatus.ERROR: {
        SessionStatus.SETTING_UP,
        SessionStatus.SETUP_DONE,
        SessionStatus.DEBATING,
        SessionStatus.READY_FOR_ADJUDICATION,
        SessionStatus.FINISHED,
    },
}


def ensure_allowed_transition(
    current: SessionStatus, target: SessionStatus
) -> SessionStatus:
    """Validate a session status transition against allowed edges.

    Args:
        current: Current session status.
        target: Desired next session status.

    Returns:
        `target` when transition is valid.

    Raises:
        ValueError: If transition `current -> target` is not allowed.
    """
    if current == target:
        return target

    allowed_targets = ALLOWED_TRANSITIONS.get(current, set())

    if target not in allowed_targets:
        raise ValueError(f"Illegal session status transition: {current} -> {target}")

    return target
