"""Main application logic for the Legal MAS GUI.

This module provides the LegalMASApp class which serves as a lightweight
facade for the underlying AppOrchestrator and its subsystems.
"""

from nicegui import ui

from apps.core.app_orchestrator import AppOrchestrator
from apps.core.chart_manager import ChartManager
from apps.core.event_handler import EventHandler
from apps.core.ui_builder import UIBuilder
from apps.core.update_manager import UpdateManager
from apps.state import AppState


class LegalMASApp:
    """Main application class for the Legal MAS GUI.

    This class serves as a lightweight facade that delegates all functionality
    to the AppOrchestrator and its subsystems (UIBuilder, EventHandler,
    UpdateManager, ChartManager).
    """

    def __init__(self, state: AppState):
        """Initialize the application.

        Args:
            state: The application state instance.
        """
        self.orchestrator = AppOrchestrator(state)
        self.orchestrator.ui_builder = UIBuilder(self.orchestrator)
        self.orchestrator.event_handler = EventHandler(self.orchestrator)
        self.orchestrator.update_manager = UpdateManager(self.orchestrator)
        self.orchestrator.chart_manager = ChartManager(self.orchestrator)
        self._on_init = self.orchestrator.event_handler._on_init
        self._on_next_turn = self.orchestrator.event_handler._on_next_turn
        self._on_node_click = self.orchestrator.event_handler._on_node_click
        self._on_timeline_change = self.orchestrator.event_handler._on_timeline_change
        self._prev_snapshot = self.orchestrator.event_handler._prev_snapshot
        self._next_snapshot = self.orchestrator.event_handler._next_snapshot
        self._exit_replay = self.orchestrator.event_handler._exit_replay
        self._reset_graph = self.orchestrator.event_handler._reset_graph
        self._toggle_left_sidebar = self.orchestrator.event_handler._toggle_left_sidebar
        self._toggle_right_panel = self.orchestrator.event_handler._toggle_right_panel
        self._show_verdict = self.orchestrator.event_handler._show_verdict
        self._update_graph = self.orchestrator.update_manager._update_graph
        self._update_transcript = self.orchestrator.update_manager._update_transcript
        self._update_sidebar = self.orchestrator.update_manager._update_sidebar

        self._update_button_states = (
            self.orchestrator.update_manager._update_button_states
        )

        self._update_timeline_slider = (
            self.orchestrator.update_manager._update_timeline_slider
        )

        self._set_loading = self.orchestrator.update_manager._set_loading
        self._refresh_all = self.orchestrator.update_manager.refresh_all
        self._resize_charts = self.orchestrator.chart_manager._resize_charts

        self._animate_chart_resize = (
            self.orchestrator.chart_manager._animate_chart_resize
        )

        self._clear_graph = self.orchestrator.update_manager._clear_graph
        self._get_active_graph_data = self.orchestrator._get_active_graph_data
        self._get_active_transcript = self.orchestrator._get_active_transcript

    def build(self):
        """Build the entire application UI.

        Registers a disconnect handler to clean up timers and resources
        when the user closes the browser tab or navigates away.
        """
        self.orchestrator.ui_builder.build()
        ui.on("disconnect", self.orchestrator.cleanup)
