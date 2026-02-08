"""Core application logic for the Legal MAS GUI.

This package contains the core application logic modules that orchestrate
the UI building, event handling, state management, and updates.
"""

from apps.core.app_orchestrator import AppOrchestrator
from apps.core.chart_manager import ChartManager
from apps.core.event_handler import EventHandler
from apps.core.ui_builder import UIBuilder
from apps.core.update_manager import UpdateManager

__all__ = [
    "AppOrchestrator",
    "UIBuilder",
    "EventHandler",
    "UpdateManager",
    "ChartManager",
]
