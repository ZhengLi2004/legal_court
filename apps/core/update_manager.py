"""Update manager for the Legal MAS GUI.

This module provides UI update logic that manages all visual updates.
"""

import logging
import traceback

from nicegui import ui

from apps.core.app_orchestrator import AppOrchestrator
from vis.adapter import EChartsAdapter

logger = logging.getLogger(__name__)


class UpdateManager:
    """Update manager that handles all UI updates.

    This class is responsible for updating the graph, transcript, sidebar,
    timeline slider, and button states. It receives an AppOrchestrator instance
    to access state and components.
    """

    def __init__(self, orchestrator: AppOrchestrator):
        """Initialize the update manager.

        Args:
            orchestrator: The application orchestrator instance.
        """
        self.orchestrator = orchestrator

    def refresh_all(self):
        """Update all UI components and controls explicitly.

        Increments the view revision counter and triggers a full refresh
        of all visual components. Called after blocking operations complete.
        """
        self.orchestrator.state.ui_state.view_revision += 1
        self._update_graph()
        self._update_transcript()
        self._update_sidebar()
        self._update_timeline_slider()
        self._update_button_states()

    def _update_button_states(self):
        """Update button enabled/disabled states based on current state.

        Considers initialization status, finish state, and replay mode. Also
        controls the verdict button visibility, which becomes available only
        when a judgment document exists.
        """
        is_init = self.orchestrator.state.engine.graph is not None
        is_finished = self.orchestrator.state.engine.is_finished
        in_replay = self.orchestrator.state.ui_state.replay_mode
        has_judgment = bool(self.orchestrator.state.engine.judgment_document)

        if is_init:
            self.orchestrator.init_btn.props("disable")

        else:
            self.orchestrator.init_btn.props(remove="disable")

        self.orchestrator.init_btn.update()

        if is_init and not is_finished and not in_replay:
            self.orchestrator.next_btn.props(remove="disable")

        else:
            self.orchestrator.next_btn.props("disable")

        self.orchestrator.next_btn.update()

        if getattr(self.orchestrator, "verdict_btn", None) is not None:
            self.orchestrator.verdict_btn.visible = has_judgment

            if has_judgment:
                self.orchestrator.verdict_btn.props(remove="disable")

            else:
                self.orchestrator.verdict_btn.props("disable")

            self.orchestrator.verdict_btn.update()

    def _update_graph(self):
        """Update the graph visualization using the active data source.

        In replay mode, renders the snapshot payload. The snapshot payload can be
        either a graph object (parsed by EChartsAdapter) or an ECharts option dict
        (applied directly). This method also provides graceful fallback behavior
        when snapshot graph payloads are missing or cannot be parsed.

        This prevents the graph view from going blank when the user scrubs the
        timeline.
        """
        if not self.orchestrator.graph_chart:
            return

        graph_payload = self.orchestrator._get_active_graph_data()

        if graph_payload is None:
            msg = (
                "回放图谱缺失"
                if self.orchestrator.state.ui_state.replay_mode
                else "无图谱数据"
            )

            self.orchestrator.graph_chart.run_chart_method(
                "setOption",
                {
                    "title": {
                        "text": msg,
                        "left": "center",
                        "top": "center",
                        "textStyle": {"color": "#999"},
                    },
                    "series": [],
                },
                {"notMerge": True},
            )

            return

        def _looks_like_echarts_option(obj: object) -> bool:
            """Check whether an object looks like a valid ECharts option dict.

            Args:
                obj: The object to check.

            Returns:
                True if the object is a dict containing typical ECharts option keys.
            """
            if not isinstance(obj, dict):
                return False

            return "series" in obj and isinstance(obj.get("series"), list)

        try:
            if _looks_like_echarts_option(graph_payload):
                option = graph_payload

            else:
                preferred = None

                if not self.orchestrator.state.ui_state.replay_mode:
                    preferred = self.orchestrator.state.engine.preferred_extension

                option = EChartsAdapter.parse_graph(
                    graph_payload,
                    preferred_extension=preferred,
                )

            self.orchestrator.graph_chart.run_chart_method(
                "setOption", option, {"notMerge": True}
            )

            self.orchestrator.chart_manager._resize_charts()

        except Exception as e:
            print(f"[ERROR] Graph update error: {e}")
            traceback.print_exc()
            live_graph = self.orchestrator.state.engine.graph

            if self.orchestrator.state.ui_state.replay_mode and live_graph is not None:
                try:
                    option = EChartsAdapter.parse_graph(
                        live_graph,
                        preferred_extension=self.orchestrator.state.engine.preferred_extension,
                    )

                    option = dict(option) if isinstance(option, dict) else option

                    if isinstance(option, dict):
                        option.setdefault("title", {})

                        option["title"].update(
                            {
                                "text": "回放解析失败（显示实时图谱）",
                                "left": "center",
                                "top": 10,
                                "textStyle": {"color": "#999", "fontSize": 12},
                            }
                        )

                    self.orchestrator.graph_chart.run_chart_method(
                        "setOption", option, {"notMerge": True}
                    )

                    self.orchestrator.chart_manager._resize_charts()
                    return

                except Exception:
                    pass

            self._clear_graph()

    def _clear_graph(self):
        """Clear the graph visualization."""
        if self.orchestrator.graph_chart:
            self.orchestrator.graph_chart.run_chart_method(
                "setOption",
                {
                    "title": {
                        "text": "无图谱数据",
                        "left": "center",
                        "top": "center",
                        "textStyle": {"color": "#999"},
                    },
                    "series": [],
                },
                {"notMerge": True},
            )

    def _update_transcript(self):
        """Update the transcript view using the active data source.

        If the right panel is hidden, skips the expensive UI update but records
        a dirty flag. When the panel becomes visible again, a refresh will be
        triggered to synchronize the view.
        """
        if not self.orchestrator.state.ui_state.right_panel_visible:
            self.orchestrator._transcript_dirty = True
            return

        lines = self.orchestrator._get_active_transcript()

        if self.orchestrator.transcript_view:
            self.orchestrator.transcript_view.refresh(lines)

        if self.orchestrator.transcript_count:
            self.orchestrator.transcript_count.set_text(str(len(lines)))
            self.orchestrator.transcript_count.update()

        self.orchestrator._transcript_dirty = False

    def _update_sidebar(self):
        """Update all sidebar cards, skipping invisible panels for performance.

        When the left sidebar is hidden, marks a dirty flag so the sidebar will
        be refreshed immediately after it becomes visible again.
        """
        if not self.orchestrator.state.ui_state.left_sidebar_visible:
            self.orchestrator._sidebar_dirty = True
            return

        if self.orchestrator.status_card:
            self.orchestrator.status_card.refresh()

        if self.orchestrator.agent_state_card:
            self.orchestrator.agent_state_card.refresh()

        if self.orchestrator.stats_card:
            self.orchestrator.stats_card.refresh()

        if self.orchestrator.convergence_chart:
            container = getattr(self.orchestrator, "convergence_container", None)

            if container is None or container.visible:
                self.orchestrator.convergence_chart.refresh()

        self.orchestrator._sidebar_dirty = False

    def _update_timeline_slider(self) -> None:
        """Update the timeline slider range based on available snapshots.

        Synchronizes slider min/max and (in live mode) moves the thumb to the
        latest snapshot. Uses a deterministic suppression flag to prevent
        the slider's on_change handler from triggering unintended replay
        mode transitions.
        """
        if not self.orchestrator.timeline_slider:
            return

        snapshots = self.orchestrator.state.engine.round_snapshots or []
        total = len(snapshots)

        self.orchestrator._suppress_timeline_events = True

        try:
            if total <= 0:
                self.orchestrator.timeline_slider._props["min"] = 0
                self.orchestrator.timeline_slider._props["max"] = 0
                self.orchestrator.timeline_slider.value = 0
                self.orchestrator.timeline_slider.update()
                return

            last_idx = total - 1
            self.orchestrator.timeline_slider._props["min"] = 0
            self.orchestrator.timeline_slider._props["max"] = last_idx

            if not self.orchestrator.state.ui_state.replay_mode:
                self.orchestrator.timeline_slider.value = last_idx
                self.orchestrator.state.ui_state.replay_round = last_idx

                if self.orchestrator.timeline_label:
                    self.orchestrator.timeline_label.text = "实时模式"

            else:
                self.orchestrator.timeline_slider.value = min(
                    max(self.orchestrator.state.ui_state.replay_round, 0), last_idx
                )

            self.orchestrator.timeline_slider.update()

        finally:
            ui.timer(
                0.1,
                lambda: setattr(self.orchestrator, "_suppress_timeline_events", False),
                once=True,
            )

    def _set_loading(self, loading: bool, message: str = ""):
        """Set loading state.

        Args:
            loading: Whether to show loading state.
            message: The status message to display when loading.
        """
        if self.orchestrator.header_spinner:
            self.orchestrator.header_spinner.visible = loading
            self.orchestrator.header_spinner.update()

        if self.orchestrator.header_status:
            self.orchestrator.header_status.text = message if loading else "就绪"
            self.orchestrator.header_status.update()
