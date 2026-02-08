"""Legal MAS Web GUI Package.

This package provides a web-based graphical user interface for the
Legal MAS debate simulation system using NiceGUI.
"""

from apps.app import LegalMASApp
from apps.components import (
    AgentStateCard,
    ConvergenceChart,
    JudgmentPreviewCard,
    NodeDetailsPanel,
    StatsCard,
    StatusCard,
    TranscriptView,
)
from apps.state import AppState, UIState, create_app_state

__all__ = [
    "UIState",
    "AppState",
    "create_app_state",
    "LegalMASApp",
    "TranscriptView",
    "StatusCard",
    "AgentStateCard",
    "StatsCard",
    "JudgmentPreviewCard",
    "NodeDetailsPanel",
    "ConvergenceChart",
]
