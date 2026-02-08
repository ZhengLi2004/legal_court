"""Application orchestrator for the Legal MAS GUI.

This module provides the core application orchestration logic that coordinates
the UI building, event handling, state management, and updates.
"""

import asyncio
from typing import Optional

from nicegui import ui

from apps.components import (
    AgentStateCard,
    ConvergenceChart,
    JudgmentPreviewCard,
    NodeDetailsPanel,
    StatsCard,
    StatusCard,
    TranscriptView,
)
from apps.state import AppState


class AppOrchestrator:
    """Core application orchestrator that coordinates all subsystems.

    This class serves as the central coordinator for the Legal MAS GUI,
    managing the application state, components, and orchestrating interactions
    between different subsystems.
    """

    def __init__(self, state: AppState):
        """Initialize the application orchestrator.

        Args:
            state: The application state instance.
        """
        self.state = state
        self.status_card: Optional[StatusCard] = None
        self.agent_state_card: Optional[AgentStateCard] = None
        self.stats_card: Optional[StatsCard] = None
        self.judgment_card: Optional[JudgmentPreviewCard] = None
        self.transcript_view: Optional[TranscriptView] = None
        self.node_details_panel: Optional[NodeDetailsPanel] = None
        self.convergence_chart: Optional[ConvergenceChart] = None
        self.graph_chart: Optional[ui.echart] = None
        self.verdict_dialog = None
        self.verdict_content = None
        self.header_status: Optional[ui.label] = None
        self.header_spinner = None
        self.next_btn = None
        self.init_btn = None
        self.transcript_count = None
        self.timeline_slider = None
        self.timeline_label = None
        self.left_sidebar_container = None
        self.right_panel_container = None
        self._suppress_timeline_events: bool = False
        self._transcript_dirty: bool = False

    def _run_blocking_setup(self, case_dict: dict):
        """Execute the engine setup in a separate thread with its own loop.

        This helper function creates a new asyncio event loop to run the
        engine setup coroutine. This is necessary because embedding model
        loading is CPU-bound and blocks the main asyncio loop for too long,
        causing WebSocket timeouts in the UI.

        Args:
            case_dict: The dictionary containing case data for initialization.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(
                self.state.engine.setup(case_data=case_dict, verbose=True)
            )

        finally:
            loop.close()

    def _run_blocking_step(self):
        """Execute the engine step in a separate thread.

        Creates a new event loop for the thread to run the async step method,
        preventing the main UI loop from blocking during LLM inference or
        graph processing.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self.state.engine.step())

        finally:
            loop.close()

    def _run_blocking_adjudicate(self):
        """Execute the adjudication process in a separate thread.

        Similar to step, this runs the final judgment generation in isolation
        to maintain UI responsiveness.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self.state.engine.adjudicate())

        finally:
            loop.close()

    def _get_active_graph_data(self):
        """Get the graph data for the current view mode.

        Returns the snapshot graph data when in replay mode,
        or the live engine graph when in live mode.

        Returns:
            The engine graph object (live mode) or a snapshot graph_data
            dict (replay mode), or None if unavailable.
        """
        if self.state.ui_state.replay_mode:
            snapshots = self.state.engine.round_snapshots
            idx = self.state.ui_state.replay_round

            if snapshots and 0 <= idx < len(snapshots):
                return snapshots[idx].get("graph_data")

            return None

        return self.state.engine.graph

    def _get_active_transcript(self):
        """Get the transcript for the current view mode.

        Returns the snapshot transcript when in replay mode,
        or the live engine transcript when in live mode.

        Returns:
            A list of transcript line strings.
        """
        if self.state.ui_state.replay_mode:
            snapshots = self.state.engine.round_snapshots
            idx = self.state.ui_state.replay_round

            if snapshots and 0 <= idx < len(snapshots):
                return snapshots[idx].get("transcript", [])

            return []

        return self.state.engine.transcript or []
