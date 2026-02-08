"""UI builder for the Legal MAS GUI.

This module provides the UI building logic that constructs all visual elements
of the application.
"""

from nicegui import ui

from apps.core.app_orchestrator import AppOrchestrator


class UIBuilder:
    """UI builder that constructs all visual elements of the application.

    This class is responsible for building the header, sidebar, main view,
    right panel, and verdict dialog. It receives an AppOrchestrator instance
    to access state and components.
    """

    def __init__(self, orchestrator: AppOrchestrator):
        """Initialize the UI builder.

        Args:
            orchestrator: The application orchestrator instance.
        """
        self.orchestrator = orchestrator

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
                overflow-x: hidden;
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

    def _build_header(self):
        """Build the header section with responsive layout.

        Includes a user-triggered verdict button which becomes available only
        after the judgment document is generated. This ensures users can stay
        on the page to inspect the graph and replay, then open the verdict
        dialog when they are ready.
        """
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
                    if self.orchestrator.state.samples:
                        options = {
                            i: f"[{i + 1}] {c.title[:25]}..."
                            for i, c in enumerate(self.orchestrator.state.samples)
                        }

                    else:
                        options = {0: "无案例数据"}

                    ui.select(
                        options,
                        value=self.orchestrator.state.selected_index,
                        label="选择案例",
                        on_change=lambda e: self.orchestrator.state.set_selected_index(
                            e.value
                        ),
                    ).classes("w-56").props("dark dense outlined")

                with ui.row().classes("items-center gap-2 flex-shrink-0"):
                    self.orchestrator.init_btn = ui.button(
                        "初始化",
                        icon="rocket",
                        on_click=self.orchestrator.event_handler._on_init,
                    ).props("color=white text-color=blue-800")

                    self.orchestrator.next_btn = ui.button(
                        "下一步",
                        icon="play_arrow",
                        on_click=self.orchestrator.event_handler._on_next_turn,
                    ).props("color=green disable")

                    self.orchestrator.verdict_btn = ui.button(
                        "宣读判决",
                        icon="gavel",
                        on_click=self.orchestrator.event_handler._show_verdict,
                    ).props("color=white text-color=blue-800")

                    self.orchestrator.verdict_btn.visible = False

                with ui.row().classes("items-center gap-2 flex-shrink-0"):
                    self.orchestrator.header_spinner = ui.spinner(
                        "dots", size="sm"
                    ).classes("text-white")

                    self.orchestrator.header_spinner.visible = False

                    self.orchestrator.header_status = ui.label("就绪").classes(
                        "text-white/80 font-mono text-sm"
                    )

                    ui.button(
                        icon="menu_open",
                        on_click=self.orchestrator.event_handler._toggle_left_sidebar,
                    ).props("flat round color=white dense").tooltip("切换左侧边栏")

                    ui.button(
                        icon="view_sidebar",
                        on_click=self.orchestrator.event_handler._toggle_right_panel,
                    ).props("flat round color=white dense").tooltip("切换右侧面板")

    def _build_left_sidebar(self):
        """Build the left sidebar with status cards.

        This sidebar contains system status, agent state, graph statistics, and
        (optionally) auxiliary visualizations such as the convergence chart.

        The judgment-related UI is intentionally removed.
        """
        self.orchestrator.left_sidebar_container = ui.element("div").classes(
            "flex-shrink-0 h-full bg-white shadow-lg panel-scroll sidebar-transition"
        )

        self.orchestrator.left_sidebar_container.style(
            f"width: {'256px' if self.orchestrator.state.ui_state.left_sidebar_visible else '0px'}"
        )

        with self.orchestrator.left_sidebar_container:
            with ui.column().classes("w-64 p-3 gap-3"):
                from apps.components import (
                    AgentStateCard,
                    ConvergenceChart,
                    StatsCard,
                    StatusCard,
                )

                self.orchestrator.status_card = StatusCard(
                    ui.element("div").classes("w-full"), self.orchestrator.state
                )

                self.orchestrator.agent_state_card = AgentStateCard(
                    ui.element("div").classes("w-full"), self.orchestrator.state
                )

                self.orchestrator.stats_card = StatsCard(
                    ui.element("div").classes("w-full"), self.orchestrator.state
                )

                with ui.card().classes("w-full p-0"):
                    with ui.row().classes(
                        "w-full items-center justify-between px-4 py-3"
                    ):
                        ui.label("📉 收敛曲线").classes(
                            "text-sm font-bold text-gray-700"
                        )
                        toggle_btn = ui.button(icon="expand_more").props(
                            "flat dense round"
                        )

                    ui.separator()

                    self.orchestrator.convergence_container = ui.element("div").classes(
                        "w-full px-3 pb-3"
                    )

                    self.orchestrator.convergence_container.visible = False

                    with self.orchestrator.convergence_container:
                        chart_host = (
                            ui.element("div")
                            .classes("w-full")
                            .style("height: clamp(140px, 20vh, 260px);")
                        )

                        self.orchestrator.convergence_chart = ConvergenceChart(
                            chart_host,
                            self.orchestrator.state,
                        )

                    def _toggle_convergence() -> None:
                        """Toggle the visibility of the convergence chart.

                        When opening the chart, refreshes chart data and forces
                        an ECharts resize so it renders correctly after being hidden.
                        """
                        is_open = not self.orchestrator.convergence_container.visible
                        self.orchestrator.convergence_container.visible = is_open
                        self.orchestrator.convergence_container.update()

                        toggle_btn.props(
                            f"icon={'expand_less' if is_open else 'expand_more'}"
                        )

                        toggle_btn.update()

                        if is_open and self.orchestrator.convergence_chart:
                            self.orchestrator.convergence_chart.refresh()
                            self.orchestrator.chart_manager._resize_charts()

                    toggle_btn.on("click", lambda: _toggle_convergence())

    def _build_center_main(self):
        """Build the center main view area.

        Keeps the graph as the primary focus. Moves the convergence chart to
        the left sidebar and leaves only a compact replay control bar here.
        """
        with ui.element("div").classes("flex-1 h-full flex flex-col p-2 gap-2 min-w-0"):
            with ui.card().classes("flex-1 flex flex-col min-h-0"):
                with ui.row().classes(
                    "justify-between items-center px-3 py-2 border-b flex-shrink-0"
                ):
                    ui.label("🔗 辩论图谱").classes("text-lg font-bold text-gray-700")

                    with ui.row().classes("gap-1"):
                        ui.button(
                            icon="center_focus_strong",
                            on_click=self.orchestrator.event_handler._reset_graph,
                        ).props("flat dense round")

                with ui.element("div").classes("flex-1 w-full min-h-0"):
                    self.orchestrator.graph_chart = (
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

                    self.orchestrator.graph_chart.on(
                        "click", self.orchestrator.event_handler._on_node_click
                    )

            with ui.card().classes("flex-shrink-0"):
                with ui.row().classes("w-full items-center gap-2 px-3 py-2"):
                    ui.label("回放").classes("text-xs font-bold text-gray-600 w-10")

                    ui.button(
                        icon="skip_previous",
                        on_click=self.orchestrator.event_handler._prev_snapshot,
                    ).props("flat dense round size=sm")

                    self.orchestrator.timeline_slider = ui.slider(
                        min=0,
                        max=0,
                        step=1,
                        value=0,
                        on_change=lambda e: (
                            self.orchestrator.event_handler._on_timeline_change(
                                int(e.value)
                            )
                        ),
                    ).classes("flex-grow")

                    ui.button(
                        icon="skip_next",
                        on_click=self.orchestrator.event_handler._next_snapshot,
                    ).props("flat dense round size=sm")

                    self.orchestrator.timeline_label = ui.label("实时模式").classes(
                        "text-xs text-gray-500 font-mono min-w-[120px] text-right"
                    )

                    self.orchestrator.live_btn = (
                        ui.button(
                            "实时",
                            on_click=self.orchestrator.event_handler._exit_replay,
                        )
                        .props("flat dense size=xs")
                        .classes("text-xs")
                    )

    def _build_right_panel(self):
        """Build the right panel with transcript and node details.

        Uses a consistent 380px width for both the outer container and
        the inner content div to prevent overflow.
        """
        self.orchestrator.right_panel_container = ui.element("div").classes(
            "flex-shrink-0 h-full bg-white shadow-lg flex flex-col sidebar-transition"
        )

        self.orchestrator.right_panel_container.style(
            f"width: {'380px' if self.orchestrator.state.ui_state.right_panel_visible else '0px'}"
        )

        with self.orchestrator.right_panel_container:
            with (
                ui.element("div").classes("flex flex-col h-full").style("width: 380px")
            ):
                with ui.row().classes(
                    "justify-between items-center px-4 py-2 border-b flex-shrink-0"
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("chat").classes("text-xl text-blue-600")
                        ui.label("对话记录").classes("text-lg font-bold text-gray-700")

                    self.orchestrator.transcript_count = ui.badge("0", color="blue")

                with ui.element("div").classes("flex-1 min-h-0 p-2"):
                    from apps.components import TranscriptView

                    self.orchestrator.transcript_view = TranscriptView(
                        ui.element("div").classes("w-full h-full"),
                        self.orchestrator.state,
                    )

                from apps.components import NodeDetailsPanel

                self.orchestrator.node_details_panel = NodeDetailsPanel(
                    ui.element("div").classes("flex-shrink-0 px-2 pb-2"),
                    self.orchestrator.state,
                )

    def _build_verdict_dialog(self):
        """Build the verdict dialog.

        The judgment preview card is removed, but the verdict dialog remains.
        The dialog HTML container uses ``sanitize=False`` because the event handler
        inserts pre-escaped content and adds only minimal structural tags (e.g., ``<br>``).

        Security:
            The event handler must HTML-escape all model-generated content before
            inserting it into this HTML container to prevent XSS.
        """
        self.orchestrator.verdict_dialog = ui.dialog().props("maximized")

        with self.orchestrator.verdict_dialog:
            with ui.card().classes("w-full h-full flex flex-col"):
                with ui.row().classes(
                    "justify-between items-center p-4 "
                    "bg-gradient-to-r from-blue-800 to-blue-600 flex-shrink-0"
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("gavel").classes("text-3xl text-white")
                        ui.label("判决宣读").classes("text-2xl font-serif text-white")

                    ui.button(
                        icon="close",
                        on_click=self.orchestrator.verdict_dialog.close,
                    ).props("flat round color=white")

                with ui.scroll_area().classes("flex-1"):
                    with ui.element("div").classes("max-w-4xl mx-auto p-8"):
                        self.orchestrator.verdict_content = ui.html(
                            "", sanitize=False
                        ).classes("font-serif text-lg leading-relaxed")
