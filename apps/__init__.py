"""Legal MAS Web GUI Package.

This package provides a web-based graphical user interface for the
Legal MAS debate simulation system using NiceGUI.
"""

from apps.state import ExecutionState, UIState, AppState, state
from apps.app import LegalMASApp
from apps.components import (
    TranscriptView,
    StatusCard,
    AgentStateCard,
    StatsCard,
    JudgmentPreviewCard,
    NodeDetailsPanel,
    ConvergenceChart,
)

__all__ = [
    "ExecutionState",
    "UIState",
    "AppState",
    "state",
    "LegalMASApp",
    "TranscriptView",
    "StatusCard",
    "AgentStateCard",
    "StatsCard",
    "JudgmentPreviewCard",
    "NodeDetailsPanel",
    "ConvergenceChart",
]