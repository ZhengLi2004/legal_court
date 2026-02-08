"""Event handler for the Legal MAS GUI.

This module provides event handling logic for all user interactions.
"""

import html
import logging

from nicegui import run, ui

from apps.core.app_orchestrator import AppOrchestrator

logger = logging.getLogger(__name__)


class EventHandler:
    """Event handler that processes all user interactions.

    This class is responsible for handling button clicks, node clicks,
    timeline changes, and other user interactions. It receives an
    AppOrchestrator instance to access state and orchestrate actions.
    """

    def __init__(self, orchestrator: AppOrchestrator):
        """Initialize the event handler.

        Args:
            orchestrator: The application orchestrator instance.
        """
        self.orchestrator = orchestrator

    async def _on_init(self):
        """Handle initialize button click.

        Acquires the state lock to prevent concurrent initialization.
        On failure, resets the engine to a clean state so the user
        can retry.
        """
        case = self.orchestrator.state.current_case

        if not case:
            ui.notify("请先选择案例", type="warning")
            return

        if self.orchestrator.state._lock.locked():
            ui.notify("操作正在进行中，请稍候", type="info")
            return

        async with self.orchestrator.state._lock:
            self.orchestrator.update_manager._set_loading(True, "初始化中...")
            self.orchestrator.init_btn.props("disable")
            self.orchestrator.init_btn.update()
            self.orchestrator.update_manager._update_button_states()

            try:
                case_dict = case.model_dump()
                await run.io_bound(self.orchestrator._run_blocking_setup, case_dict)
                self.orchestrator.state.ui_state.last_transcript_count = 0
                self.orchestrator.state.ui_state.last_node_count = 0
                self.orchestrator.state.ui_state.last_edge_count = 0
                self.orchestrator.update_manager._update_graph()
                self.orchestrator.update_manager._update_transcript()
                self.orchestrator.update_manager._update_sidebar()
                self.orchestrator.update_manager._update_button_states()
                ui.notify("✅ 初始化完成!", type="positive")

            except Exception as e:
                logger.error("Initialization failed: %s", e, exc_info=True)

                self.orchestrator.state.engine = __import__(
                    "mas.engine", fromlist=["DebateEngine"]
                ).DebateEngine(
                    config=__import__(
                        "mas.config", fromlist=["SystemConfig"]
                    ).SystemConfig(),
                    judge_config={},
                )

                ui.notify("❌ 初始化失败，请重试", type="negative")
                self.orchestrator.update_manager._update_button_states()

            finally:
                self.orchestrator.update_manager._set_loading(False)

    async def _on_next_turn(self):
        """Handle next turn button click.

        Runs the debate forward by one step. If the debate reaches completion,
        it generates the judgment document (if applicable) but does not open
        the verdict dialog automatically. Users can remain on the page to inspect
        the graph/transcript/replay and click the dedicated verdict button when
        they are ready.
        """
        if not self.orchestrator.state.is_initialized:
            ui.notify("请先初始化", type="warning")
            return

        if self.orchestrator.state.engine.is_finished:
            ui.notify("辩论已结束，可点击“宣读判决”查看", type="info")
            return

        if self.orchestrator.state._lock.locked():
            ui.notify("操作正在进行中，请稍候", type="info")
            return

        verdict_ready = False

        async with self.orchestrator.state._lock:
            if self.orchestrator.state.ui_state.replay_mode:
                self._exit_replay()

            turn = self.orchestrator.state.engine.current_turn
            turn_map = {"plaintiff": "原告", "defendant": "被告"}
            turn_name = turn_map.get(turn.value, turn.value)
            self.orchestrator.update_manager._set_loading(
                True, f"执行{turn_name}回合..."
            )
            self.orchestrator.next_btn.props("disable")
            self.orchestrator.next_btn.update()
            live_monitor = ui.timer(0.2, self.orchestrator.agent_state_card.refresh)
            self.orchestrator.register_timer(live_monitor)

            try:
                await run.io_bound(self.orchestrator._run_blocking_step)

                if (
                    self.orchestrator.state.engine.is_ready_for_adjudication
                    and not self.orchestrator.state.engine.is_finished
                ):
                    self.orchestrator.update_manager._set_loading(
                        True, "正在生成判决..."
                    )
                    await run.io_bound(self.orchestrator._run_blocking_adjudicate)

                verdict_ready = bool(self.orchestrator.state.engine.judgment_document)

                if self.orchestrator.state.engine.is_finished:
                    ui.notify("🏁 辩论结束", type="positive")

            except Exception as e:
                logger.error("Turn execution failed: %s", e, exc_info=True)
                ui.notify("❌ 执行失败，请重试", type="negative")

            finally:
                live_monitor.cancel()
                self.orchestrator.update_manager._set_loading(False)
                self.orchestrator.update_manager.refresh_all()

        if self.orchestrator.state.engine.is_finished and verdict_ready:
            ui.notify(
                "判决已生成，你可以先查看图谱/回放，然后点击“宣读判决”打开", type="info"
            )

    def _on_node_click(self, e):
        """Handle node click in the graph.

        Safely extracts the node ID from the event data, falling back
        gracefully if the expected fields are not present.

        Args:
            e: The click event object containing node data.
        """
        if not e.args:
            return

        data = (
            e.args
            if isinstance(e.args, dict)
            else e.args.get("data", {})
            if isinstance(e.args, dict)
            else {}
        )

        if isinstance(e.args, dict):
            data = e.args.get("data", e.args)

        node_id = data.get("name", "") or data.get("id", "")

        if not node_id:
            return

        if (
            self.orchestrator.state.engine.graph
            and self.orchestrator.state.engine.graph.graph.has_node(node_id)
            and self.orchestrator.node_details_panel
        ):
            self.orchestrator.node_details_panel.show_node(node_id)

    def _on_timeline_change(self, value: int) -> None:
        """Handle timeline slider change.

        User-driven slider changes enter replay mode, except when the user
        moves the slider to the latest snapshot. In that case, the UI
        automatically exits replay mode and returns to live mode.

        Programmatic slider updates are suppressed to avoid unintended mode
        switches.

        Args:
            value: The timeline slider value representing the snapshot index.
        """
        if self.orchestrator._suppress_timeline_events:
            return

        snapshots = self.orchestrator.state.engine.round_snapshots or []

        if not snapshots:
            return

        last_idx = len(snapshots) - 1
        value = max(0, min(value, last_idx))
        self.orchestrator.state.ui_state.replay_round = value

        if value == last_idx:
            self.orchestrator.state.ui_state.replay_mode = False

            if self.orchestrator.timeline_label:
                self.orchestrator.timeline_label.text = "实时模式"

            self.orchestrator.update_manager._update_graph()
            self.orchestrator.update_manager._update_transcript()
            self.orchestrator.update_manager._update_button_states()
            return

        self.orchestrator.state.ui_state.replay_mode = True
        snap = snapshots[value]

        if self.orchestrator.timeline_label:
            self.orchestrator.timeline_label.text = (
                f"回合 {snap.get('round_idx', value)} - {snap.get('turn', '')}"
            )

        self.orchestrator.update_manager._update_graph()
        self.orchestrator.update_manager._update_transcript()
        self.orchestrator.update_manager._update_button_states()

    def _prev_snapshot(self):
        """Go to previous snapshot."""
        if self.orchestrator.state.ui_state.replay_round > 0:
            new_val = self.orchestrator.state.ui_state.replay_round - 1
            self.orchestrator.timeline_slider.value = new_val
            self._on_timeline_change(new_val)

    def _next_snapshot(self):
        """Go to next snapshot."""
        max_val = (
            len(self.orchestrator.state.engine.round_snapshots) - 1
            if self.orchestrator.state.engine.round_snapshots
            else 0
        )

        if self.orchestrator.state.ui_state.replay_round < max_val:
            new_val = self.orchestrator.state.ui_state.replay_round + 1
            self.orchestrator.timeline_slider.value = new_val
            self._on_timeline_change(new_val)

    def _exit_replay(self) -> None:
        """Exit replay mode and return to live view.

        Keeps the slider position aligned with the latest snapshot through
        ``refresh_all()`` and ``_update_timeline_slider()``.
        """
        self.orchestrator.state.ui_state.replay_mode = False

        if self.orchestrator.timeline_label:
            self.orchestrator.timeline_label.text = "实时模式"

        self.orchestrator.update_manager.refresh_all()

    def _reset_graph(self):
        """Reset graph view zoom/pan to default state.

        Only dispatches a restore action without re-setting the option,
        so the reset animation is not immediately overwritten.
        """
        if self.orchestrator.graph_chart:
            self.orchestrator.graph_chart.run_chart_method(
                "dispatchAction", {"type": "restore"}
            )

    def _toggle_left_sidebar(self) -> None:
        """Toggle left sidebar visibility with a smooth transition.

        Also triggers continuous chart resizing so the center ECharts canvases
        follow the layout transition in real time. If the sidebar was hidden while
        updates occurred, it performs a refresh immediately after reopening.
        """
        self.orchestrator.state.ui_state.left_sidebar_visible = (
            not self.orchestrator.state.ui_state.left_sidebar_visible
        )

        self.orchestrator.left_sidebar_container.style(
            f"width: {'256px' if self.orchestrator.state.ui_state.left_sidebar_visible else '0px'}"
        )

        self.orchestrator.left_sidebar_container.update()
        self.orchestrator.chart_manager._animate_chart_resize()

        if self.orchestrator.state.ui_state.left_sidebar_visible and getattr(
            self.orchestrator, "_sidebar_dirty", False
        ):
            self.orchestrator.update_manager._update_sidebar()

    def _toggle_right_panel(self) -> None:
        """Toggle right panel visibility with a smooth transition.

        When the panel becomes visible, the transcript view is refreshed
        if it was previously skipped while hidden.
        """
        self.orchestrator.state.ui_state.right_panel_visible = (
            not self.orchestrator.state.ui_state.right_panel_visible
        )

        self.orchestrator.right_panel_container.style(
            f"width: {'380px' if self.orchestrator.state.ui_state.right_panel_visible else '0px'}"
        )

        self.orchestrator.right_panel_container.update()
        self.orchestrator.chart_manager._animate_chart_resize()

        if (
            self.orchestrator.state.ui_state.right_panel_visible
            and self.orchestrator._transcript_dirty
        ):
            self.orchestrator.update_manager._update_transcript()

    def _show_verdict(self):
        """Show the verdict dialog.

        Renders the judgment document into a dialog. All model-generated content is
        HTML-escaped to prevent XSS, then newlines are converted to ``<br>`` for display.
        """
        if not getattr(self.orchestrator, "verdict_dialog", None) or not getattr(
            self.orchestrator, "verdict_content", None
        ):
            ui.notify("判决对话框未初始化", type="warning")
            return

        if self.orchestrator.state.engine.judgment_document:
            raw = self.orchestrator.state.engine.judgment_document
            safe = html.escape(raw).replace("\n", "<br>")

            self.orchestrator.verdict_content.content = (
                f"<div class='text-justify'>{safe}</div>"
            )

        else:
            self.orchestrator.verdict_content.content = (
                "<p class='text-gray-500 text-center py-8'>判决内容尚未生成</p>"
            )

        self.orchestrator.verdict_dialog.open()
