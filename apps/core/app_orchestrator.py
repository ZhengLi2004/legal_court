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

    Attributes:
        state: The application state instance.
        ui_builder: The UI builder subsystem.
        event_handler: The event handler subsystem.
        update_manager: The update manager subsystem.
        chart_manager: The chart manager subsystem.
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
        self.convergence_container = None
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
        self._sidebar_dirty: bool = False
        self._active_timers: list = []

    def _run_blocking_setup(self, case_dict: dict):
        """Execute the engine setup in a separate thread with its own loop.

        Creates a temporary asyncio event loop for the current thread without
        polluting the thread-local event loop state. This prevents issues when
        the thread pool worker is reused by subsequent tasks.

        Args:
            case_dict: The dictionary containing case data for initialization.
        """
        loop = asyncio.new_event_loop()

        try:
            loop.run_until_complete(
                self.state.engine.setup(case_data=case_dict, verbose=True)
            )

        finally:
            loop.close()

    def _run_blocking_step(self):
        """Execute the engine step in a separate thread.

        Creates a temporary event loop without setting it as the thread default,
        preventing stale loop references in pooled threads.
        """
        loop = asyncio.new_event_loop()

        try:
            loop.run_until_complete(self.state.engine.step())

        finally:
            loop.close()

    def _run_blocking_adjudicate(self):
        """Execute the adjudication process in a separate thread.

        Creates a temporary event loop without setting it as the thread default,
        preventing stale loop references in pooled threads.
        """
        loop = asyncio.new_event_loop()

        try:
            loop.run_until_complete(self.state.engine.adjudicate())

        finally:
            loop.close()

    def register_timer(self, timer) -> None:
        """Register a timer for cleanup when the session ends.

        Args:
            timer: The NiceGUI timer instance to track.
        """
        self._active_timers.append(timer)

    def cleanup(self) -> None:
        """Cancel all active timers and release resources.

        Should be called when the user disconnects or navigates away
        to prevent memory leaks from orphaned timers.
        """
        for timer in self._active_timers:
            try:
                timer.cancel()

            except Exception:
                pass

        self._active_timers.clear()
        resize_timer = getattr(self, "_chart_resize_timer", None)

        if resize_timer:
            try:
                resize_timer.cancel()

            except Exception:
                pass

            self._chart_resize_timer = None

        stop_timer = getattr(self, "_chart_resize_stop_timer", None)

        if stop_timer:
            try:
                stop_timer.cancel()

            except Exception:
                pass

            self._chart_resize_stop_timer = None

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
