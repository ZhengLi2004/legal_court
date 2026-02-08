"""Event handler for the Legal MAS GUI.

This module provides event handling logic for all user interactions.
"""

import traceback

from nicegui import run, ui

from apps.core.app_orchestrator import AppOrchestrator


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
        """Handle initialize button click."""
        case = self.orchestrator.state.current_case

        if not case:
            ui.notify("请先选择案例", type="warning")
            return

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
            traceback.print_exc()
            ui.notify(f"❌ 初始化失败: {e}", type="negative")
            self.orchestrator.update_manager._update_button_states()

        finally:
            self.orchestrator.update_manager._set_loading(False)

    async def _on_next_turn(self):
        """Handle next turn button click.

        If in replay mode, automatically exits replay before proceeding.
        Disables the next button during execution to prevent double-clicks.
        """
        if not self.orchestrator.state.is_initialized:
            ui.notify("请先初始化", type="warning")
            return

        if self.orchestrator.state.engine.is_finished:
            ui.notify("辩论已结束", type="info")
            self._show_verdict()
            return

        if self.orchestrator.state.ui_state.replay_mode:
            self._exit_replay()

        turn = self.orchestrator.state.engine.current_turn
        turn_map = {"plaintiff": "原告", "defendant": "被告"}
        turn_name = turn_map.get(turn.value, turn.value)
        self.orchestrator.update_manager._set_loading(True, f"执行{turn_name}回合...")
        self.orchestrator.next_btn.props("disable")
        self.orchestrator.next_btn.update()
        live_monitor = ui.timer(0.2, self.orchestrator.agent_state_card.refresh)

        try:
            await run.io_bound(self.orchestrator._run_blocking_step)

            if (
                self.orchestrator.state.engine.is_ready_for_adjudication
                and not self.orchestrator.state.engine.is_finished
            ):
                self.orchestrator.update_manager._set_loading(True, "正在生成判决...")
                await run.io_bound(self.orchestrator._run_blocking_adjudicate)

            if self.orchestrator.state.engine.is_finished:
                ui.notify("🏁 辩论结束，判决已生成!", type="positive")

        except Exception as e:
            print(f"[ERROR] Turn failed: {e}")
            traceback.print_exc()
            ui.notify(f"❌ 执行失败: {e}", type="negative")

        finally:
            live_monitor.cancel()
            self.orchestrator.update_manager._set_loading(False)
            self.orchestrator.update_manager.refresh_all()

    def _on_node_click(self, e):
        """Handle node click in the graph.

        Args:
            e: The click event object containing node data.
        """
        if not e.args:
            return

        data = e.args.get("data", {})
        node_id = data.get("name") or data.get("id", "")

        if node_id and self.orchestrator.node_details_panel:
            self.orchestrator.node_details_panel.show_node(node_id)

    def _on_timeline_change(self, value: int) -> None:
        """Handle timeline slider change.

        User-driven slider changes enter replay mode, except when the user moves
        the slider to the latest snapshot. In that case, the UI automatically
        exits replay mode and returns to live mode.

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

        if value < 0 or value > last_idx:
            return

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
        `_refresh_all()` and `_update_timeline_slider()`.
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
        follow the layout transition in real time.
        """
        self.orchestrator.state.ui_state.left_sidebar_visible = (
            not self.orchestrator.state.ui_state.left_sidebar_visible
        )

        self.orchestrator.left_sidebar_container.style(
            f"width: {'256px' if self.orchestrator.state.ui_state.left_sidebar_visible else '0px'}"
        )

        self.orchestrator.left_sidebar_container.update()
        self.orchestrator.chart_manager._animate_chart_resize()

    def _toggle_right_panel(self) -> None:
        """Toggle right panel visibility with a smooth transition.

        When the panel becomes visible, the transcript view is refreshed if it was
        previously skipped while hidden.
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
        """Show the verdict dialog."""
        if self.orchestrator.state.engine.judgment_document:
            content = self.orchestrator.state.engine.judgment_document.replace(
                "\n", "<br>"
            )
            self.orchestrator.verdict_content.content = (
                f"<div class='text-justify'>{content}</div>"
            )
        else:
            self.orchestrator.verdict_content.content = (
                "<p class='text-gray-500 text-center py-8'>判决书尚未生成</p>"
            )

        self.orchestrator.verdict_dialog.open()
