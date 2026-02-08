"""Main application logic for the Legal MAS GUI.

This module provides the LegalMASApp class which orchestrates
the entire UI building, event handling, and state coordination.
"""

import asyncio
import traceback
from typing import Optional

from nicegui import ui, run
from vis.adapter import EChartsAdapter

from apps.state import AppState
from apps.components import (
    TranscriptView,
    StatusCard,
    AgentStateCard,
    StatsCard,
    JudgmentPreviewCard,
    NodeDetailsPanel,
    ConvergenceChart,
)


class LegalMASApp:
    """Main application class for the Legal MAS GUI."""

    def __init__(self, state: AppState):
        """Initialize the application.

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
        self.graph_chart: Optional[ui.echart] = None
        self.convergence_chart: Optional[ConvergenceChart] = None
        self.verdict_dialog = None
        self.verdict_content = None
        self.auto_run_timer = None
        self.header_status: Optional[ui.label] = None
        self.header_spinner = None
        self.next_btn = None
        self.init_btn = None
        self.transcript_count = None
        self.timeline_slider = None
        self.timeline_label = None
        self.left_sidebar_container = None
        self._suppress_timeline_events: bool = False
        self._transcript_dirty: bool = False

    def build(self):
        """Build the entire application UI."""
        ui.colors(primary="#1E40AF", secondary="#7C3AED", accent="#10B981")

        ui.add_head_html(
            """
        <style>
            :root { --header-height: 56px; }
            html, body { height: 100%; margin: 0; }
            .main-layout {
                height: calc(100vh - var(--header-height));
                min-height: 500px;
                overflow: hidden;
            }
            .panel-scroll { overflow-y: auto; overflow-x: hidden; }
            .sidebar-transition {
                transition: width 0.25s ease-in-out;
                overflow: hidden;
            }
        </style>
        """
        )

        self._build_header()

        with ui.element("div").classes("w-full main-layout flex bg-gray-100"):
            self._build_left_sidebar()
            self._build_center_main()
            self._build_right_panel()

        self._build_verdict_dialog()

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

    def _build_header(self):
        """Build the header section with responsive layout."""
        with (
            ui.header()
            .classes(
                "bg-gradient-to-r from-blue-800 to-blue-600 px-4 py-2 shadow-lg items-center"
            )
            .style("height: var(--header-height)")
        ):
            with ui.row().classes(
                "w-full items-center justify-between flex-nowrap gap-3"
            ):
                with ui.row().classes("items-center gap-2 flex-shrink-0"):
                    ui.icon("balance").classes("text-2xl text-white")
                    ui.label("Legal MAS").classes("text-lg font-bold text-white")

                with ui.row().classes("items-center gap-2 flex-shrink min-w-0"):
                    if self.state.samples:
                        options = {
                            i: f"[{i + 1}] {c.title[:25]}..."
                            for i, c in enumerate(self.state.samples)
                        }

                    else:
                        options = {0: "无案例数据"}

                    ui.select(
                        options,
                        value=self.state.selected_index,
                        label="选择案例",
                        on_change=lambda e: setattr(
                            self.state, "selected_index", e.value
                        ),
                    ).classes("w-56").props("dark dense outlined")

                with ui.row().classes("items-center gap-2 flex-shrink-0"):
                    self.init_btn = ui.button(
                        "初始化", icon="rocket", on_click=self._on_init
                    ).props("color=white text-color=blue-800")

                    self.next_btn = ui.button(
                        "下一步", icon="play_arrow", on_click=self._on_next_turn
                    ).props("color=green disable")

                with ui.row().classes("items-center gap-2 flex-shrink-0"):
                    self.header_spinner = ui.spinner("dots", size="sm").classes(
                        "text-white"
                    )

                    self.header_spinner.visible = False

                    self.header_status = ui.label("就绪").classes(
                        "text-white/80 font-mono text-sm"
                    )

                    ui.button(
                        icon="menu_open", on_click=self._toggle_left_sidebar
                    ).props("flat round color=white dense").tooltip("切换左侧边栏")

                    ui.button(
                        icon="view_sidebar", on_click=self._toggle_right_panel
                    ).props("flat round color=white dense").tooltip("切换右侧面板")

    def _build_left_sidebar(self):
        """Build the left sidebar with status cards."""
        self.left_sidebar_container = ui.element("div").classes(
            "flex-shrink-0 h-full bg-white shadow-lg panel-scroll sidebar-transition"
        )

        self.left_sidebar_container.style(
            f"width: {'256px' if self.state.ui_state.left_sidebar_visible else '0px'}"
        )

        with self.left_sidebar_container:
            with ui.column().classes("w-64 p-3 gap-3"):
                self.status_card = StatusCard(
                    ui.element("div").classes("w-full"), self.state
                )

                self.agent_state_card = AgentStateCard(
                    ui.element("div").classes("w-full"), self.state
                )

                self.stats_card = StatsCard(
                    ui.element("div").classes("w-full"), self.state
                )

                self.judgment_card = JudgmentPreviewCard(
                    ui.element("div").classes("w-full"),
                    self.state,
                    on_view_full=self._show_verdict,
                )

    def _build_center_main(self):
        """Build the center main view area."""
        with ui.element("div").classes("flex-1 h-full flex flex-col p-2 gap-2 min-w-0"):
            with ui.card().classes("flex-1 flex flex-col min-h-0"):
                with ui.row().classes(
                    "justify-between items-center px-3 py-2 border-b flex-shrink-0"
                ):
                    ui.label("🔗 辩论图谱").classes("text-lg font-bold text-gray-700")

                    with ui.row().classes("gap-1"):
                        ui.button(
                            icon="center_focus_strong", on_click=self._reset_graph
                        ).props("flat dense round")

                with ui.element("div").classes("flex-1 w-full min-h-0"):
                    self.graph_chart = (
                        ui.echart(
                            {
                                "title": {
                                    "text": "等待初始化...",
                                    "left": "center",
                                    "top": "center",
                                    "textStyle": {"color": "#999", "fontSize": 16},
                                },
                                "series": [],
                            }
                        )
                        .classes("w-full h-full")
                        .style("min-height: 300px")
                    )

                    self.graph_chart.on("click", self._on_node_click)

            with ui.card().classes("h-28 flex-shrink-0"):
                with ui.row().classes("w-full h-full gap-2 p-1"):
                    with ui.element("div").classes("flex-1 h-full"):
                        self.convergence_chart = ConvergenceChart(
                            ui.element("div").classes("w-full h-full"), self.state
                        )

                    with ui.column().classes("w-48 h-full justify-center gap-1 px-2"):
                        ui.label("回放控制").classes("text-xs font-bold text-gray-600")

                        with ui.row().classes("w-full items-center gap-1"):
                            ui.button(
                                icon="skip_previous", on_click=self._prev_snapshot
                            ).props("flat dense round size=sm")

                            self.timeline_slider = ui.slider(
                                min=0,
                                max=0,
                                step=1,
                                value=0,
                                on_change=lambda e: self._on_timeline_change(
                                    int(e.value)
                                ),
                            ).classes("flex-grow")

                            ui.button(
                                icon="skip_next", on_click=self._next_snapshot
                            ).props("flat dense round size=sm")

                        with ui.row().classes("w-full items-center justify-between"):
                            self.timeline_label = ui.label("实时模式").classes(
                                "text-xs text-gray-500 font-mono text-center"
                            )

                            self.live_btn = (
                                ui.button("实时", on_click=self._exit_replay)
                                .props("flat dense size=xs")
                                .classes("text-xs")
                            )

    def _build_right_panel(self):
        """Build the right panel with transcript and node details."""
        self.right_panel_container = ui.element("div").classes(
            "flex-shrink-0 h-full bg-white shadow-lg flex flex-col sidebar-transition"
        )

        self.right_panel_container.style(
            f"width: {'380px' if self.state.ui_state.right_panel_visible else '0px'}"
        )

        with self.right_panel_container:
            with ui.element("div").classes("w-96 flex flex-col h-full"):
                with ui.row().classes(
                    "justify-between items-center px-4 py-2 border-b flex-shrink-0"
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("chat").classes("text-xl text-blue-600")
                        ui.label("对话记录").classes("text-lg font-bold text-gray-700")

                    self.transcript_count = ui.badge("0", color="blue")

                with ui.element("div").classes("flex-1 min-h-0 p-2"):
                    self.transcript_view = TranscriptView(
                        ui.element("div").classes("w-full h-full"), self.state
                    )

                self.node_details_panel = NodeDetailsPanel(
                    ui.element("div").classes("flex-shrink-0 px-2 pb-2"), self.state
                )

    def _build_verdict_dialog(self):
        """Build the verdict dialog."""
        self.verdict_dialog = ui.dialog().props("maximized")

        with self.verdict_dialog:
            with ui.card().classes("w-full h-full flex flex-col"):
                with ui.row().classes(
                    "justify-between items-center p-4 "
                    "bg-gradient-to-r from-blue-800 to-blue-600 flex-shrink-0"
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("gavel").classes("text-3xl text-white")
                        ui.label("民事判决书").classes("text-2xl font-serif text-white")

                    ui.button(icon="close", on_click=self.verdict_dialog.close).props(
                        "flat round color=white"
                    )

                with ui.scroll_area().classes("flex-1"):
                    with ui.element("div").classes("max-w-4xl mx-auto p-8"):
                        self.verdict_content = ui.html("", sanitize=False).classes(
                            "font-serif text-lg leading-relaxed"
                        )

    def _update_button_states(self):
        """Update button enabled/disabled states based on current state.

        Considers initialization status, engine running state, finish state,
        and whether the UI is currently in replay mode.
        """
        is_init = self.state.engine.graph is not None
        is_finished = self.state.engine.is_finished
        in_replay = self.state.ui_state.replay_mode

        if is_init:
            self.init_btn.props("disable")

        else:
            self.init_btn.props(remove="disable")

        self.init_btn.update()

        if is_init and not is_finished and not in_replay:
            self.next_btn.props(remove="disable")

        else:
            self.next_btn.props("disable")

        self.next_btn.update()

    async def _on_init(self):
        """Handle initialize button click."""
        case = self.state.current_case

        if not case:
            ui.notify("请先选择案例", type="warning")
            return

        self._set_loading(True, "初始化中...")
        self.init_btn.props("disable")
        self.init_btn.update()
        self._update_button_states()

        try:
            case_dict = case.model_dump()
            await run.io_bound(self._run_blocking_setup, case_dict)
            self.state.ui_state.last_transcript_count = 0
            self.state.ui_state.last_node_count = 0
            self.state.ui_state.last_edge_count = 0
            self._update_graph()
            self._update_transcript()
            self._update_sidebar()
            self._update_button_states()
            ui.notify("✅ 初始化完成!", type="positive")

        except Exception as e:
            traceback.print_exc()
            ui.notify(f"❌ 初始化失败: {e}", type="negative")
            self._update_button_states()

        finally:
            self._set_loading(False)

    async def _on_next_turn(self):
        """Handle next turn button click.

        If in replay mode, automatically exits replay before proceeding.
        Disables the next button during execution to prevent double-clicks.
        """
        if not self.state.is_initialized:
            ui.notify("请先初始化", type="warning")
            return

        if self.state.engine.is_finished:
            ui.notify("辩论已结束", type="info")
            self._show_verdict()
            return

        if self.state.ui_state.replay_mode:
            self._exit_replay()

        turn = self.state.engine.current_turn
        turn_map = {"plaintiff": "原告", "defendant": "被告"}
        turn_name = turn_map.get(turn.value, turn.value)
        self._set_loading(True, f"执行{turn_name}回合...")
        self.next_btn.props("disable")
        self.next_btn.update()
        live_monitor = ui.timer(0.2, self.agent_state_card.refresh)

        try:
            await run.io_bound(self._run_blocking_step)

            if (
                self.state.engine.is_ready_for_adjudication
                and not self.state.engine.is_finished
            ):
                self._set_loading(True, "正在生成判决...")
                await run.io_bound(self._run_blocking_adjudicate)

            if self.state.engine.is_finished:
                ui.notify("🏁 辩论结束，判决已生成!", type="positive")

        except Exception as e:
            print(f"[ERROR] Turn failed: {e}")
            traceback.print_exc()
            ui.notify(f"❌ 执行失败: {e}", type="negative")

        finally:
            live_monitor.cancel()
            self._set_loading(False)
            self._refresh_all()

    def _on_node_click(self, e):
        """Handle node click in the graph.

        Args:
            e: The click event object containing node data.
        """
        if not e.args:
            return

        data = e.args.get("data", {})
        node_id = data.get("name") or data.get("id", "")

        if node_id and self.node_details_panel:
            self.node_details_panel.show_node(node_id)

    def _apply_snapshot(self, snapshot: dict):
        """Apply a snapshot to update the graph display.

        Args:
            snapshot: A dictionary containing snapshot data including graph_data.
        """
        if not snapshot:
            return

        graph_data = snapshot.get("graph_data", {})
        nodes = graph_data.get("nodes", [])
        links = graph_data.get("edges", [])

        if not nodes:
            return

        echart_nodes = []
        echart_links = []
        category_color_map = {}

        for node in nodes:
            n_id = node.get("id", "")
            n_data = {k: v for k, v in node.items() if k != "id"}

            cat_name, color, size, border_color, border_width = (
                EChartsAdapter._get_category_name_and_color(n_data)
            )

            if cat_name not in category_color_map:
                category_color_map[cat_name] = color

            content = str(n_data.get("content", ""))[:100]

            echart_nodes.append(
                {
                    "id": str(n_id),
                    "name": str(n_id),
                    "value": content,
                    "symbolSize": size,
                    "itemStyle": {
                        "color": color,
                        "borderColor": border_color,
                        "borderWidth": border_width,
                    },
                    "category": cat_name,
                }
            )

        for edge in links:
            source = edge.get("source", "")
            target = edge.get("target", "")

            e_type = EChartsAdapter._extract_enum_value(
                edge.get("type", "SUPPORT")
            ).upper()

            is_support = e_type == "SUPPORT"

            echart_links.append(
                {
                    "source": str(source),
                    "target": str(target),
                    "lineStyle": {
                        "color": "#32CD32" if is_support else "#FF4500",
                        "width": 2,
                        "type": "solid" if is_support else "dashed",
                    },
                }
            )

        categories_list = [
            {"name": name, "itemStyle": {"color": color}}
            for name, color in sorted(category_color_map.items())
        ]

        option = {
            "animation": False,
            "series": [
                {
                    "type": "graph",
                    "layout": "force",
                    "data": echart_nodes,
                    "links": echart_links,
                    "categories": categories_list,
                    "roam": True,
                    "force": {
                        "repulsion": 350,
                        "layoutAnimation": False,
                    },
                }
            ],
        }

        self.graph_chart.run_chart_method("setOption", option, {"notMerge": False})

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
        if self._suppress_timeline_events:
            return

        snapshots = self.state.engine.round_snapshots or []

        if not snapshots:
            return

        last_idx = len(snapshots) - 1

        if value < 0 or value > last_idx:
            return

        self.state.ui_state.replay_round = value

        if value == last_idx:
            self.state.ui_state.replay_mode = False

            if self.timeline_label:
                self.timeline_label.text = "实时模式"

            self._update_graph()
            self._update_transcript()
            self._update_button_states()
            return

        self.state.ui_state.replay_mode = True
        snap = snapshots[value]

        if self.timeline_label:
            self.timeline_label.text = (
                f"回合 {snap.get('round_idx', value)} - {snap.get('turn', '')}"
            )

        self._update_graph()
        self._update_transcript()
        self._update_button_states()

    def _prev_snapshot(self):
        """Go to previous snapshot."""
        if self.state.ui_state.replay_round > 0:
            new_val = self.state.ui_state.replay_round - 1
            self.timeline_slider.value = new_val
            self._on_timeline_change(new_val)

    def _next_snapshot(self):
        """Go to next snapshot."""
        max_val = (
            len(self.state.engine.round_snapshots) - 1
            if self.state.engine.round_snapshots
            else 0
        )

        if self.state.ui_state.replay_round < max_val:
            new_val = self.state.ui_state.replay_round + 1
            self.timeline_slider.value = new_val
            self._on_timeline_change(new_val)

    def _exit_replay(self) -> None:
        """Exit replay mode and return to live view.

        Keeps the slider position aligned with the latest snapshot through
        `_refresh_all()` and `_update_timeline_slider()`.
        """
        self.state.ui_state.replay_mode = False

        if self.timeline_label:
            self.timeline_label.text = "实时模式"

        self._refresh_all()

    def _reset_graph(self):
        """Reset graph view zoom/pan to default state.

        Only dispatches a restore action without re-setting the option,
        so the reset animation is not immediately overwritten.
        """
        if self.graph_chart:
            self.graph_chart.run_chart_method("dispatchAction", {"type": "restore"})

    def _toggle_left_sidebar(self) -> None:
        """Toggle left sidebar visibility with a smooth transition.

        Also triggers continuous chart resizing so the center ECharts canvases
        follow the layout transition in real time.
        """
        self.state.ui_state.left_sidebar_visible = (
            not self.state.ui_state.left_sidebar_visible
        )

        self.left_sidebar_container.style(
            f"width: {'256px' if self.state.ui_state.left_sidebar_visible else '0px'}"
        )

        self.left_sidebar_container.update()
        self._animate_chart_resize()

    def _resize_charts(self) -> None:
        """Resize all ECharts instances once.

        This method is used for non-animated layout changes or post-render
        corrections (e.g., after setOption). It keeps the implementation
        centralized so callers do not need to know which charts exist.
        """
        if self.graph_chart:
            self.graph_chart.run_chart_method("resize")

        if self.convergence_chart and getattr(self.convergence_chart, "chart", None):
            self.convergence_chart.chart.run_chart_method("resize")

    def _animate_chart_resize(
        self,
        duration: float = 0.28,
        interval: float = 0.05,
    ) -> None:
        """Continuously resize charts during a CSS width transition.

        Side panels expand/collapse with a CSS width transition. During this time,
        the center area width changes continuously. ECharts does not automatically
        follow container size changes, so the canvas can appear to lag or overlap
        with adjacent panels.

        This method starts a short-lived timer to call `resize()` repeatedly during
        the transition and performs one final resize at the end.

        Args:
            duration: Total seconds to keep resizing. Should be slightly larger
                than the CSS transition duration (e.g., 0.25s -> 0.28s).
            interval: Resize cadence in seconds (e.g., 0.05s).
        """
        prev_timer = getattr(self, "_chart_resize_timer", None)

        if prev_timer:
            prev_timer.cancel()

        prev_stop = getattr(self, "_chart_resize_stop_timer", None)

        if prev_stop:
            prev_stop.cancel()

        self._resize_charts()
        self._chart_resize_timer = ui.timer(interval, self._resize_charts)

        def _stop_and_finalize() -> None:
            timer = getattr(self, "_chart_resize_timer", None)

            if timer:
                timer.cancel()
                self._chart_resize_timer = None

            self._resize_charts()

        self._chart_resize_stop_timer = ui.timer(
            duration, _stop_and_finalize, once=True
        )

    def _toggle_right_panel(self) -> None:
        """Toggle right panel visibility with a smooth transition.

        When the panel becomes visible, the transcript view is refreshed if it was
        previously skipped while hidden.
        """
        self.state.ui_state.right_panel_visible = (
            not self.state.ui_state.right_panel_visible
        )

        self.right_panel_container.style(
            f"width: {'380px' if self.state.ui_state.right_panel_visible else '0px'}"
        )

        self.right_panel_container.update()
        self._animate_chart_resize()

        if self.state.ui_state.right_panel_visible and self._transcript_dirty:
            self._update_transcript()

    def _show_verdict(self):
        """Show the verdict dialog."""
        if self.state.engine.judgment_document:
            content = self.state.engine.judgment_document.replace("\n", "<br>")
            self.verdict_content.content = f"<div class='text-justify'>{content}</div>"

        else:
            self.verdict_content.content = (
                "<p class='text-gray-500 text-center py-8'>判决书尚未生成</p>"
            )

        self.verdict_dialog.open()

    def _set_loading(self, loading: bool, message: str = ""):
        """Set loading state.

        Args:
            loading: Whether to show loading state.
            message: The status message to display when loading.
        """
        if self.header_spinner:
            self.header_spinner.visible = loading
            self.header_spinner.update()

        if self.header_status:
            self.header_status.text = message if loading else "就绪"
            self.header_status.update()

    def _refresh_all(self):
        """Update all UI components and controls explicitly.

        Increments the view revision counter and triggers a full refresh
        of all visual components. Called after blocking operations complete.
        """
        self.state.ui_state.view_revision += 1
        self._update_graph()
        self._update_transcript()
        self._update_sidebar()
        self._update_timeline_slider()
        self._update_button_states()

    def _update_timeline_slider(self) -> None:
        """Update the timeline slider range based on available snapshots.

        Synchronizes slider min/max and (in live mode) moves the thumb to the
        latest snapshot without triggering replay mode.
        """
        if not self.timeline_slider:
            return

        snapshots = self.state.engine.round_snapshots or []
        total = len(snapshots)

        if total <= 0:
            self._suppress_timeline_events = True
            self.timeline_slider._props["min"] = 0
            self.timeline_slider._props["max"] = 0
            self.timeline_slider.value = 0
            self.timeline_slider.update()

            ui.timer(
                0.05,
                lambda: setattr(self, "_suppress_timeline_events", False),
                once=True,
            )
            return

        last_idx = total - 1
        self._suppress_timeline_events = True
        self.timeline_slider._props["min"] = 0
        self.timeline_slider._props["max"] = last_idx

        if not self.state.ui_state.replay_mode:
            self.timeline_slider.value = last_idx
            self.state.ui_state.replay_round = last_idx

            if self.timeline_label:
                self.timeline_label.text = "实时模式"

        else:
            self.timeline_slider.value = min(
                max(self.state.ui_state.replay_round, 0), last_idx
            )

        self.timeline_slider.update()

        ui.timer(
            0.05, lambda: setattr(self, "_suppress_timeline_events", False), once=True
        )

    def _update_graph(self):
        """Update the graph visualization."""
        if not self.graph_chart:
            return

        if not self.state.engine.graph:
            self._clear_graph()
            return

        try:
            option = EChartsAdapter.parse_graph(
                self.state.engine.graph,
                preferred_extension=self.state.engine.preferred_extension,
            )

            self.graph_chart.run_chart_method("setOption", option, {"notMerge": True})
            self._resize_charts()

        except Exception as e:
            print(f"[ERROR] Graph update error: {e}")
            traceback.print_exc()

    def _clear_graph(self):
        """Clear the graph visualization."""
        if self.graph_chart:
            self.graph_chart.run_chart_method(
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
        if not self.state.ui_state.right_panel_visible:
            self._transcript_dirty = True
            return

        lines = self._get_active_transcript()

        if self.transcript_view:
            self.transcript_view.refresh(lines)

        if self.transcript_count:
            self.transcript_count.set_text(str(len(lines)))
            self.transcript_count.update()

        self._transcript_dirty = False

    def _update_sidebar(self):
        """Update all sidebar cards, skipping invisible panels for performance."""
        if self.state.ui_state.left_sidebar_visible:
            if self.status_card:
                self.status_card.refresh()

            if self.agent_state_card:
                self.agent_state_card.refresh()

            if self.stats_card:
                self.stats_card.refresh()

            if self.judgment_card:
                self.judgment_card.refresh()

        if self.convergence_chart:
            self.convergence_chart.refresh()

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
