"""UI Components for the Legal MAS GUI.

This package contains all UI component classes for the application,
including views, cards, panels, and charts. Each component receives
state through dependency injection.
"""

from apps.components.cards import (
    AgentStateCard,
    JudgmentPreviewCard,
    StatsCard,
    StatusCard,
)
from apps.components.charts import ConvergenceChart
from apps.components.panels import NodeDetailsPanel
from apps.components.views import TranscriptView

__all__ = [
    "TranscriptView",
    "StatusCard",
    "AgentStateCard",
    "StatsCard",
    "JudgmentPreviewCard",
    "NodeDetailsPanel",
    "ConvergenceChart",
]
