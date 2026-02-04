"""Creates a web-based graphical user interface (GUI) for the Legal MAS.

This module uses the NiceGUI library to build an interactive web front end for
the debate simulation.
"""

import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from nicegui import ui

from data.loader import CaseDataLoader
from mas.config import SystemConfig
from mas.engine import DebateEngine
from vis.adapter import EChartsAdapter


class ExecutionState(Enum):
    """Represents the current execution state of the application."""

    IDLE = auto()
    INITIALIZING = auto()
    RUNNING_TURN = auto()
    AUTO_RUNNING = auto()
    FINISHED = auto()


@dataclass
class UIState:
    """Centralized UI state management."""

    execution_state: ExecutionState = ExecutionState.IDLE
    selected_node_id: Optional[str] = None
    auto_run_enabled: bool = False
    auto_run_interval: float = 2.5
    show_node_details: bool = False
    current_status_message: str = "Ready"
    replay_mode: bool = False
    replay_round: int = 0
    last_transcript_count: int = 0
    last_node_count: int = 0
    last_edge_count: int = 0


@dataclass
class AppState:
    """Manage the global state of the web application."""

    engine: DebateEngine = field(
        default_factory=lambda: DebateEngine(config=SystemConfig(), judge_config={})
    )

    loader: CaseDataLoader = field(
        default_factory=lambda: CaseDataLoader("data/sampling")
    )

    samples: list = field(default_factory=list)
    selected_index: int = 0
    ui_state: UIState = field(default_factory=UIState)

    def __post_init__(self):
        """Load samples after initialization."""
        try:
            self.samples = self.loader.load_all(limit=20)

        except Exception as e:
            print(f"Warning: Failed to load samples: {e}")
            self.samples = []

    @property
    def current_case(self):
        """Get the currently selected CaseData object."""
        if self.samples and 0 <= self.selected_index < len(self.samples):
            return self.samples[self.selected_index]

        return None

    @property
    def is_initialized(self) -> bool:
        """Check if the engine has been initialized."""
        return self.engine.graph is not None

    def reset(self):
        """Reset the engine and UI state."""
        self.engine = DebateEngine(config=SystemConfig(), judge_config={})
        self.ui_state = UIState()


state = AppState()


class TranscriptView:
    """A scrollable transcript view with auto-scroll capability."""

    def __init__(self, parent):
        """Initialize the TranscriptView component."""
        with parent:
            self.scroll_area = ui.scroll_area().classes(
                "w-full h-full bg-slate-50 rounded-lg"
            )

            with self.scroll_area:
                self.container = ui.column().classes("w-full gap-2 p-3")

        self._last_count = 0

    def refresh(self, lines: list[str]):
        """Update the transcript with new lines.

        Args:
            lines: A list of transcript lines to display.
        """
        self._last_count = len(lines)
        self.container.clear()

        with self.container:
            if not lines:
                ui.label("等待辩论开始...").classes(
                    "text-gray-500 italic text-center py-8"
                )

                return

            for i, line in enumerate(lines):
                self._render_line(line, i)

        self.scroll_area.scroll_to(percent=1.0)

    def _render_line(self, line: str, index: int):
        """Render a single transcript line.

        Args:
            line: The transcript line text to render.
            index: The index of the line in the transcript.
        """
        bg_colors = ["bg-white", "bg-gray-50"]
        bg = bg_colors[index % 2]

        with ui.card().classes(f"w-full p-3 {bg} shadow-sm"):
            if "【" in line and "】" in line:
                parts = line.split("】", 1)
                speaker = parts[0] + "】"
                content = parts[1].strip() if len(parts) > 1 else ""

                if "原告" in speaker or "Plaintiff" in speaker:
                    icon_name = "person"
                    color = "text-blue-600"

                elif "被告" in speaker or "Defendant" in speaker:
                    icon_name = "person_outline"
                    color = "text-red-600"

                elif "法官" in speaker or "Judge" in speaker:
                    icon_name = "gavel"
                    color = "text-purple-600"

                elif "系统" in speaker or "System" in speaker:
                    icon_name = "settings"
                    color = "text-gray-600"

                else:
                    icon_name = "chat"
                    color = "text-gray-700"

                with ui.row().classes("items-start gap-2 w-full"):
                    ui.icon(icon_name).classes(f"text-xl {color} flex-shrink-0")

                    with ui.column().classes("flex-1 gap-1 min-w-0"):
                        ui.label(speaker).classes(f"font-bold text-sm {color}")
                        ui.label(content).classes(
                            "text-sm text-gray-700 break-words whitespace-pre-wrap"
                        )

            else:
                ui.label(line).classes(
                    "text-sm text-gray-700 break-words whitespace-pre-wrap"
                )


class StatusCard:
    """A card displaying current system status."""

    def __init__(self, parent):
        """Initialize the StatusCard component."""
        with parent:
            with ui.card().classes("w-full p-4"):
                ui.label("📊 系统状态").classes("text-lg font-bold text-gray-700 mb-2")
                ui.separator()

                with ui.column().classes("w-full gap-2 mt-3"):
                    with ui.row().classes("justify-between items-center"):
                        ui.label("回合:").classes("text-sm text-gray-600")

                        self.round_label = ui.label("0").classes(
                            "font-mono font-bold text-blue-600"
                        )

                    with ui.row().classes("justify-between items-center"):
                        ui.label("当前方:").classes("text-sm text-gray-600")
                        self.turn_label = ui.label("就绪").classes("font-mono")

                    with ui.row().classes("justify-between items-center"):
                        ui.label("收敛度:").classes("text-sm text-gray-600")

                        self.conv_label = ui.label("0.0000").classes(
                            "font-mono text-green-600"
                        )

                    ui.separator().classes("my-1")

                    with ui.row().classes("justify-between items-center"):
                        ui.label("执行状态:").classes("text-sm text-gray-600")

                        self.state_badge = ui.badge("IDLE", color="gray").classes(
                            "px-2"
                        )

    def refresh(self):
        """Update the status card with current engine state."""
        self.round_label.text = str(state.engine.round_idx)
        turn = state.engine.current_turn

        if turn:
            turn_map = {"plaintiff": "原告", "defendant": "被告"}
            turn_text = turn_map.get(turn.value, turn.value.capitalize())
            self.turn_label.text = turn_text
            color_map = {"plaintiff": "text-blue-600", "defendant": "text-red-600"}

            self.turn_label.classes(
                replace=f"font-mono {color_map.get(turn.value, 'text-gray-600')}"
            )

        else:
            self.turn_label.text = "就绪"
            self.turn_label.classes(replace="font-mono text-gray-600")

        conv = 0.0

        if state.engine.convergence_history:
            conv = state.engine.convergence_history[-1]

        self.conv_label.text = f"{conv:.4f}"
        exec_state = state.ui_state.execution_state

        state_map = {
            ExecutionState.IDLE: ("空闲", "gray"),
            ExecutionState.INITIALIZING: ("初始化中", "yellow"),
            ExecutionState.RUNNING_TURN: ("执行中", "blue"),
            ExecutionState.AUTO_RUNNING: ("自动运行", "green"),
            ExecutionState.FINISHED: ("已完成", "purple"),
        }

        text, color = state_map.get(exec_state, ("未知", "gray"))
        self.state_badge.set_text(text)
        self.state_badge._props["color"] = color
        self.state_badge.update()


class AgentStateCard:
    """A card displaying agent state machine status."""

    def __init__(self, parent):
        """Initialize the AgentStateCard component."""
        with parent:
            with ui.card().classes("w-full p-4"):
                ui.label("🤖 代理状态").classes("text-lg font-bold text-gray-700 mb-2")
                ui.separator()
                self.content = ui.column().classes("w-full gap-2 mt-3")

                with self.content:
                    self._render_placeholder()

    def _render_placeholder(self):
        """Render placeholder content."""
        ui.label("等待系统初始化...").classes("text-sm text-gray-500 italic py-2")

    def _get_state_color(self, state_name: str) -> str:
        """Get the color class for a given state.

        Args:
            state_name: The name of the agent state.

        Returns:
            The CSS color class string for the state.
        """
        colors = {
            "IDLE": "bg-gray-200 text-gray-700",
            "ASSESS_NEEDS": "bg-yellow-200 text-yellow-800",
            "WAIT_WORKERS": "bg-orange-200 text-orange-800",
            "INGEST_RESULTS": "bg-blue-200 text-blue-800",
            "DECIDE_ACTION": "bg-indigo-200 text-indigo-800",
            "DONE": "bg-green-200 text-green-800",
        }
        return colors.get(state_name, "bg-gray-200 text-gray-700")

    def refresh(self):
        """Update the agent state display."""
        self.content.clear()

        is_init = (
            state.engine.graph is not None
            and state.engine.p_team is not None
            and state.engine.d_team is not None
        )

        if not is_init:
            with self.content:
                self._render_placeholder()

            return

        p_state = "IDLE"
        d_state = "IDLE"

        try:
            if state.engine.p_team:
                p_state = state.engine.p_team.pipeline_step.name

            if state.engine.d_team:
                d_state = state.engine.d_team.pipeline_step.name

        except Exception as e:
            print(f"Error getting agent state: {e}")

        with self.content:
            with ui.row().classes("items-center gap-2 w-full"):
                ui.icon("person", color="blue").classes("text-lg")
                ui.label("原告:").classes("text-sm font-medium w-12")

                ui.label(p_state).classes(
                    f"px-2 py-1 rounded text-xs font-bold flex-1 text-center "
                    f"{self._get_state_color(p_state)}"
                )

            with ui.row().classes("items-center gap-2 w-full"):
                ui.icon("person_outline", color="red").classes("text-lg")
                ui.label("被告:").classes("text-sm font-medium w-12")

                ui.label(d_state).classes(
                    f"px-2 py-1 rounded text-xs font-bold flex-1 text-center "
                    f"{self._get_state_color(d_state)}"
                )

        self.content.update()


class StatsCard:
    """A card displaying graph statistics."""

    def __init__(self, parent):
        """Initialize the StatsCard component."""
        with parent:
            with ui.card().classes("w-full p-4"):
                ui.label("📈 图谱统计").classes("text-lg font-bold text-gray-700 mb-2")
                ui.separator()
                self.content = ui.column().classes("w-full gap-1 mt-3")

                with self.content:
                    self._render_empty()

    def _render_empty(self):
        """Render empty state."""
        stats = [
            ("节点", "0", "bubble_chart"),
            ("边", "0", "timeline"),
            ("诉求", "0", "gavel"),
            ("事实", "0", "fact_check"),
            ("法条", "0", "menu_book"),
        ]

        for label, value, icon in stats:
            self._render_stat_row(label, value, icon)

    def _render_stat_row(self, label: str, value: str, icon: str):
        """Render a single stat row.

        Args:
            label: The label text for the stat.
            value: The value text to display.
            icon: The icon name for the stat.
        """
        with ui.row().classes("justify-between items-center w-full py-0.5"):
            with ui.row().classes("items-center gap-1"):
                ui.icon(icon).classes("text-sm text-gray-400")
                ui.label(label).classes("text-sm text-gray-600")

            ui.label(value).classes("font-mono text-sm font-bold text-gray-800")

    def refresh(self):
        """Update statistics from the current graph."""
        self.content.clear()

        if not state.engine.graph:
            with self.content:
                self._render_empty()

            return

        try:
            G = state.engine.graph.graph

            def get_type_str(d):
                t = d.get("type")

                if hasattr(t, "value"):
                    return t.value

                if hasattr(t, "name"):
                    return t.name

                return str(t) if t else ""

            claim_count = sum(
                1 for _, d in G.nodes(data=True) if get_type_str(d).upper() == "CLAIM"
            )

            fact_count = sum(
                1 for _, d in G.nodes(data=True) if get_type_str(d).upper() == "FACT"
            )

            law_count = sum(
                1 for _, d in G.nodes(data=True) if get_type_str(d).upper() == "LAW"
            )

            support_count = sum(
                1
                for _, _, d in G.edges(data=True)
                if get_type_str(d).upper() == "SUPPORT"
            )

            conflict_count = sum(
                1
                for _, _, d in G.edges(data=True)
                if get_type_str(d).upper() in ("CONFLICT", "ATTACK")
            )

            with self.content:
                self._render_stat_row("节点", str(G.number_of_nodes()), "bubble_chart")
                self._render_stat_row("边", str(G.number_of_edges()), "timeline")
                ui.separator().classes("my-1")
                self._render_stat_row("诉求", str(claim_count), "gavel")
                self._render_stat_row("事实", str(fact_count), "fact_check")
                self._render_stat_row("法条", str(law_count), "menu_book")
                ui.separator().classes("my-1")
                self._render_stat_row("支持边", str(support_count), "thumb_up")
                self._render_stat_row("冲突边", str(conflict_count), "flash_on")

            self.content.update()

        except Exception as e:
            with self.content:
                ui.label(f"统计错误: {e}").classes("text-red-500 text-sm")


class JudgmentPreviewCard:
    """A card showing judgment preview and status."""

    def __init__(self, parent, on_view_full):
        """Initialize the JudgmentPreviewCard.

        Args:
            parent: The parent UI element.
            on_view_full: Callback function to show full verdict dialog.
        """
        self.on_view_full = on_view_full
        self.phase_indicators = {}

        with parent:
            with ui.card().classes("w-full p-4"):
                with ui.row().classes("justify-between items-center mb-2"):
                    ui.label("⚖️ 判决").classes("text-lg font-bold text-gray-700")

                    self.view_btn = (
                        ui.button("查看", icon="open_in_new", on_click=on_view_full)
                        .props("flat dense")
                        .classes("text-xs")
                    )

                    self.view_btn.visible = False

                ui.separator()
                self.content = ui.column().classes("w-full gap-2 mt-3")

                with self.content:
                    self._render_phases()

    def _render_phases(self):
        """Render the three-phase indicators."""
        phases = [
            ("LLM", "psychology", "blue"),
            ("BAF", "account_tree", "green"),
            ("融合", "merge_type", "purple"),
        ]

        with ui.row().classes("justify-around w-full py-2"):
            for name, icon, color in phases:
                with ui.column().classes("items-center gap-1"):
                    ui.icon(icon).classes(f"text-2xl text-{color}-400")
                    ui.label(name).classes("text-xs text-gray-600")
                    indicator = ui.label("⏳").classes("text-lg")
                    self.phase_indicators[name] = indicator

    def refresh(self):
        """Update the judgment preview."""
        is_finished = state.engine.is_finished
        has_judgment = bool(state.engine.judgment_document)
        self.view_btn.visible = has_judgment
        self.view_btn.update()

        for _, indicator in self.phase_indicators.items():
            if is_finished:
                indicator.text = "✅"

            elif state.ui_state.execution_state == ExecutionState.RUNNING_TURN:
                indicator.text = "🔄"

            else:
                indicator.text = "⏳"


class NodeDetailsPanel:
    """A panel showing details of the selected node."""

    def __init__(self, parent):
        """Initialize the NodeDetailsPanel.

        Args:
            parent: The parent UI element.
        """
        with parent:
            self.card = ui.card().classes("w-full p-4 mt-2 border-t")
            self.card.visible = False

            with self.card:
                with ui.row().classes("justify-between items-center mb-2"):
                    ui.label("🔍 节点详情").classes("text-lg font-bold text-gray-700")

                    ui.button(icon="close", on_click=self._on_close).props(
                        "flat dense round"
                    )

                ui.separator()
                self.content = ui.column().classes("w-full gap-2 mt-2")

    def _on_close(self):
        """Handle close button click."""
        state.ui_state.show_node_details = False
        state.ui_state.selected_node_id = None
        self.card.visible = False
        self.card.update()

    def show_node(self, node_id: str):
        """Display details for the specified node.

        Args:
            node_id: The ID of the node to display.
        """
        if not state.engine.graph:
            return

        if not state.engine.graph.graph.has_node(node_id):
            return

        state.ui_state.selected_node_id = node_id
        state.ui_state.show_node_details = True
        self.card.visible = True
        node_data = state.engine.graph.graph.nodes[node_id]
        self.content.clear()

        with self.content:
            ui.label(f"ID: {node_id}").classes("font-mono text-xs text-gray-500")
            n_type = node_data.get("type", "N/A")

            if hasattr(n_type, "value"):
                n_type = n_type.value

            n_type = str(n_type).upper()

            type_styles = {
                "CLAIM": ("bg-blue-100 text-blue-800", "gavel"),
                "FACT": ("bg-green-100 text-green-800", "fact_check"),
                "LAW": ("bg-orange-100 text-orange-800", "menu_book"),
            }

            style, icon = type_styles.get(n_type, ("bg-gray-100 text-gray-800", "help"))

            with ui.row().classes("items-center gap-2"):
                ui.icon(icon).classes("text-lg")
                ui.label("类型:").classes("text-sm font-medium")

                ui.label(n_type).classes(
                    f"px-2 py-0.5 rounded text-xs font-bold {style}"
                )

            ui.label("内容:").classes("text-sm font-medium mt-2")
            content = str(node_data.get("content", "N/A"))
            display_content = content[:400] + "..." if len(content) > 400 else content

            ui.label(display_content).classes(
                "text-sm text-gray-700 break-words bg-gray-50 p-2 rounded"
            )

            with ui.row().classes("gap-4 mt-2"):
                status = node_data.get("status", "N/A")

                if hasattr(status, "value"):
                    status = status.value

                with ui.column().classes("gap-0"):
                    ui.label("状态").classes("text-xs text-gray-500")
                    ui.label(str(status)).classes("text-sm font-mono")

                agent_id = str(node_data.get("agent_id", "N/A"))

                with ui.column().classes("gap-0"):
                    ui.label("代理").classes("text-xs text-gray-500")
                    ui.label(agent_id[:15]).classes("text-sm font-mono")

            ui.separator().classes("my-2")
            in_edges = list(state.engine.graph.graph.in_edges(node_id))
            out_edges = list(state.engine.graph.graph.out_edges(node_id))

            ui.label(f"连接: {len(in_edges)} 入 | {len(out_edges)} 出").classes(
                "text-sm text-gray-600"
            )

        self.card.update()


class ConvergenceChart:
    """A component for the convergence curve chart."""

    def __init__(self, parent):
        """Initialize the ConvergenceChart.

        Args:
            parent: The parent UI element.
        """
        with parent:
            self.chart = ui.echart(
                {
                    "title": {
                        "text": "收敛曲线",
                        "left": "center",
                        "top": 5,
                        "textStyle": {"fontSize": 12},
                    },
                    "grid": {"top": 35, "right": 15, "bottom": 25, "left": 45},
                    "xAxis": {"type": "category", "data": []},
                    "yAxis": {"type": "value", "min": 0},
                    "series": [{"type": "line", "data": [], "smooth": True}],
                }
            ).classes("w-full h-full")

    def refresh(self):
        """Update the convergence chart with latest data."""
        history = state.engine.convergence_history or []
        rounds = list(range(1, len(history) + 1))

        option = {
            "title": {
                "text": "收敛曲线",
                "left": "center",
                "top": 5,
                "textStyle": {"fontSize": 12},
            },
            "grid": {"top": 35, "right": 15, "bottom": 25, "left": 45},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "category", "data": rounds, "axisLabel": {"fontSize": 9}},
            "yAxis": {"type": "value", "min": 0, "axisLabel": {"fontSize": 9}},
            "series": [
                {
                    "type": "line",
                    "data": history,
                    "smooth": True,
                    "showSymbol": len(history) < 30,
                    "symbolSize": 5,
                    "itemStyle": {"color": "#3B82F6"},
                    "lineStyle": {"width": 2},
                    "areaStyle": {
                        "color": {
                            "type": "linear",
                            "x": 0,
                            "y": 0,
                            "x2": 0,
                            "y2": 1,
                            "colorStops": [
                                {"offset": 0, "color": "rgba(59,130,246,0.3)"},
                                {"offset": 1, "color": "rgba(59,130,246,0.05)"},
                            ],
                        }
                    },
                }
            ],
        }

        self.chart.run_chart_method("setOption", option, {"notMerge": True})


class LegalMASApp:
    """Main application class for the Legal MAS GUI."""

    def __init__(self):
        """Initialize the application."""
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
        self.poll_timer = None
        self.header_status: Optional[ui.label] = None
        self.header_spinner = None
        self.auto_btn = None
        self.next_btn = None
        self.init_btn = None
        self.transcript_count = None
        self.timeline_slider = None
        self.timeline_label = None

    def build(self):
        """Build the entire application UI."""
        ui.colors(primary="#1E40AF", secondary="#7C3AED", accent="#10B981")

        ui.add_head_html("""
        <style>
            :root { --header-height: 56px; }
            html, body { height: 100%; margin: 0; overflow: hidden; }
            .main-layout {
                height: calc(100vh - var(--header-height));
                min-height: 500px;
            }
            .panel-scroll { overflow-y: auto; overflow-x: hidden; }
        </style>
        """)

        self._build_header()

        with ui.element("div").classes("w-full main-layout flex bg-gray-100"):
            self._build_left_sidebar()
            self._build_center_main()
            self._build_right_panel()

        self._build_verdict_dialog()
        self.poll_timer = ui.timer(1.0, self._poll_updates)

    def _build_header(self):
        """Build the header section."""
        with (
            ui.header()
            .classes(
                "bg-gradient-to-r from-blue-800 to-blue-600 px-4 py-2 shadow-lg items-center"
            )
            .style("height: var(--header-height)")
        ):
            with ui.row().classes("w-full items-center justify-between"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("balance").classes("text-2xl text-white")
                    ui.label("Legal MAS").classes("text-lg font-bold text-white")

                with ui.row().classes("items-center gap-2"):
                    if state.samples:
                        options = {
                            i: f"[{i + 1}] {c.title[:25]}..."
                            for i, c in enumerate(state.samples)
                        }

                    else:
                        options = {0: "无案例数据"}

                    ui.select(
                        options,
                        value=state.selected_index,
                        label="选择案例",
                        on_change=lambda e: setattr(state, "selected_index", e.value),
                    ).classes("w-56").props("dark dense outlined")

                with ui.row().classes("items-center gap-2"):
                    self.init_btn = ui.button(
                        "初始化", icon="rocket", on_click=self._on_init
                    ).props("color=white text-color=blue-800")

                    self.next_btn = ui.button(
                        "下一步", icon="play_arrow", on_click=self._on_next_turn
                    ).props("color=green disable")

                    self.auto_btn = ui.button(
                        "自动", icon="autorenew", on_click=self._toggle_auto_run
                    ).props("color=amber disable")

                    ui.button("重置", icon="refresh", on_click=self._on_reset).props(
                        "color=red flat"
                    )

                with ui.row().classes("items-center gap-2"):
                    self.header_spinner = ui.spinner("dots", size="sm").classes(
                        "text-white"
                    )

                    self.header_spinner.visible = False

                    self.header_status = ui.label("就绪").classes(
                        "text-white/80 font-mono text-sm"
                    )

    def _build_left_sidebar(self):
        """Build the left sidebar with status cards."""
        with ui.element("div").classes(
            "w-64 flex-shrink-0 h-full bg-white shadow-lg panel-scroll"
        ):
            with ui.column().classes("w-full p-3 gap-3"):
                self.status_card = StatusCard(ui.element("div").classes("w-full"))

                self.agent_state_card = AgentStateCard(
                    ui.element("div").classes("w-full")
                )

                self.stats_card = StatsCard(ui.element("div").classes("w-full"))

                self.judgment_card = JudgmentPreviewCard(
                    ui.element("div").classes("w-full"), on_view_full=self._show_verdict
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
                        ui.button(icon="zoom_in", on_click=self._zoom_in).props(
                            "flat dense round"
                        )

                        ui.button(icon="zoom_out", on_click=self._zoom_out).props(
                            "flat dense round"
                        )

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
                            ui.element("div").classes("w-full h-full")
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
        with ui.element("div").classes(
            "w-[380px] flex-shrink-0 h-full bg-white shadow-lg flex flex-col"
        ):
            with ui.row().classes(
                "justify-between items-center px-4 py-2 border-b flex-shrink-0"
            ):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("chat").classes("text-xl text-blue-600")
                    ui.label("对话记录").classes("text-lg font-bold text-gray-700")

                self.transcript_count = ui.badge("0", color="blue")

            with ui.element("div").classes("flex-1 min-h-0 p-2"):
                self.transcript_view = TranscriptView(
                    ui.element("div").classes("w-full h-full")
                )

            self.node_details_panel = NodeDetailsPanel(
                ui.element("div").classes("flex-shrink-0 px-2 pb-2")
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
        """Update button enabled/disabled states based on current state."""
        is_init = state.engine.graph is not None
        is_finished = state.engine.is_finished

        is_running = state.ui_state.execution_state in (
            ExecutionState.INITIALIZING,
            ExecutionState.RUNNING_TURN,
        )

        is_auto = state.ui_state.auto_run_enabled

        if is_running:
            self.init_btn.props("disable")

        else:
            self.init_btn.props(remove="disable")

        self.init_btn.update()

        if is_init and not is_finished and not is_running and not is_auto:
            self.next_btn.props(remove="disable")

        else:
            self.next_btn.props("disable")

        self.next_btn.update()

        if is_init and not is_finished and not is_running:
            self.auto_btn.props(remove="disable")

        else:
            if not is_auto:
                self.auto_btn.props("disable")

        self.auto_btn.update()

    async def _on_init(self):
        """Handle initialize button click."""
        case = state.current_case

        if not case:
            ui.notify("请先选择案例", type="warning")
            return

        state.ui_state.execution_state = ExecutionState.INITIALIZING
        self._set_loading(True, "初始化中...")
        self._update_button_states()

        try:
            case_dict = case.model_dump()
            print("[DEBUG] Starting engine setup...")
            await state.engine.setup(case_data=case_dict, verbose=True)

            print(
                f"[DEBUG] Engine setup complete. Graph: {state.engine.graph is not None}"
            )

            state.ui_state.execution_state = ExecutionState.IDLE
            state.ui_state.last_transcript_count = 0
            state.ui_state.last_node_count = 0
            state.ui_state.last_edge_count = 0
            self._update_graph()
            self._update_transcript()
            self._update_sidebar()
            self._update_button_states()
            ui.notify("✅ 初始化完成!", type="positive")

        except Exception as e:
            print(f"[ERROR] Init failed: {e}")
            traceback.print_exc()
            ui.notify(f"❌ 初始化失败: {e}", type="negative")
            state.ui_state.execution_state = ExecutionState.IDLE
            self._update_button_states()

        finally:
            self._set_loading(False)

    async def _on_next_turn(self):
        """Handle next turn button click."""
        if not state.is_initialized:
            ui.notify("请先初始化", type="warning")
            return

        if state.engine.is_finished:
            ui.notify("辩论已结束", type="info")
            self._show_verdict()
            return

        turn = state.engine.current_turn
        turn_map = {"plaintiff": "原告", "defendant": "被告"}
        turn_name = turn_map.get(turn.value, turn.value)
        state.ui_state.execution_state = ExecutionState.RUNNING_TURN
        self._set_loading(True, f"执行{turn_name}回合...")
        self._update_button_states()

        try:
            await state.engine.step()

            if state.engine.is_ready_for_adjudication and not state.engine.is_finished:
                self._set_loading(True, "正在生成判决...")
                await state.engine.adjudicate()

            if state.engine.is_finished:
                state.ui_state.execution_state = ExecutionState.FINISHED
                ui.notify("🏁 辩论结束，判决已生成!", type="positive")

            else:
                state.ui_state.execution_state = ExecutionState.IDLE

            self._refresh_all()
            self._update_button_states()

        except Exception as e:
            print(f"[ERROR] Turn failed: {e}")
            traceback.print_exc()
            ui.notify(f"❌ 执行失败: {e}", type="negative")
            state.ui_state.execution_state = ExecutionState.IDLE
            self._update_button_states()

        finally:
            self._set_loading(False)

    def _toggle_auto_run(self):
        """Toggle auto-run mode."""
        if state.ui_state.auto_run_enabled:
            state.ui_state.auto_run_enabled = False

            if self.auto_run_timer:
                self.auto_run_timer.cancel()
                self.auto_run_timer = None

            self.auto_btn.props("color=amber")
            self.auto_btn.text = "自动"
            state.ui_state.execution_state = ExecutionState.IDLE
            self._update_button_states()
            ui.notify("⏹ 自动运行已停止", type="info")

        else:
            if not state.is_initialized:
                ui.notify("请先初始化", type="warning")
                return

            if state.engine.is_finished:
                ui.notify("辩论已结束", type="info")
                return

            state.ui_state.auto_run_enabled = True
            state.ui_state.execution_state = ExecutionState.AUTO_RUNNING
            self.auto_btn.props("color=red")
            self.auto_btn.text = "停止"
            self._update_button_states()
            ui.notify("▶ 自动运行已启动", type="positive")

            self.auto_run_timer = ui.timer(
                state.ui_state.auto_run_interval, self._auto_run_step
            )

    async def _auto_run_step(self):
        """Execute one step in auto-run mode."""
        if not state.ui_state.auto_run_enabled:
            return

        if state.engine.is_finished:
            self._toggle_auto_run()
            return

        await self._on_next_turn()

    def _on_reset(self):
        """Handle reset button click."""
        if state.ui_state.auto_run_enabled:
            self._toggle_auto_run()

        state.reset()
        state.ui_state = UIState()
        self._refresh_all()
        self._clear_graph()
        self.auto_btn.props("color=amber")
        self.auto_btn.text = "自动"
        self._update_button_states()
        ui.notify("🔄 系统已重置", type="warning")

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

        from vis.adapter import EChartsAdapter

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

    def _on_timeline_change(self, value: int):
        """Handle timeline slider change.

        Args:
            value: The timeline slider value representing the snapshot index.
        """
        snapshots = state.engine.round_snapshots

        if not snapshots:
            return

        if 0 <= value < len(snapshots):
            state.ui_state.replay_mode = True
            state.ui_state.replay_round = value
            snap = snapshots[value]

            self.timeline_label.text = (
                f"回合 {snap.get('round_idx', value)} - {snap.get('turn', '')}"
            )

            self._apply_snapshot(snap)
            snap_transcript = snap.get("transcript", [])

            if self.transcript_view:
                self.transcript_view._last_count = -1  # 强制刷新
                self.transcript_view.refresh(snap_transcript)

    def _prev_snapshot(self):
        """Go to previous snapshot."""
        if state.ui_state.replay_round > 0:
            new_val = state.ui_state.replay_round - 1
            self.timeline_slider.value = new_val
            self._on_timeline_change(new_val)

    def _next_snapshot(self):
        """Go to next snapshot."""
        max_val = (
            len(state.engine.round_snapshots) - 1 if state.engine.round_snapshots else 0
        )

        if state.ui_state.replay_round < max_val:
            new_val = state.ui_state.replay_round + 1
            self.timeline_slider.value = new_val
            self._on_timeline_change(new_val)

    def _exit_replay(self):
        """Exit replay mode and return to live view."""
        state.ui_state.replay_mode = False
        state.ui_state.replay_round = 0
        self.timeline_label.text = "实时模式"
        state.ui_state.last_node_count = -1
        state.ui_state.last_edge_count = -1
        state.ui_state.last_transcript_count = -1
        self._update_graph()
        self._update_transcript()
        self._update_sidebar()

    def _poll_updates(self):
        """Poll for state changes and update UI."""
        self._update_button_states()

        if state.ui_state.replay_mode:
            return

        if not state.is_initialized:
            return

        current_transcript = (
            len(state.engine.transcript) if state.engine.transcript else 0
        )

        if current_transcript != state.ui_state.last_transcript_count:
            state.ui_state.last_transcript_count = current_transcript
            self._update_transcript()
            self._update_sidebar()

        if state.engine.graph:
            current_nodes = state.engine.graph.graph.number_of_nodes()
            current_edges = state.engine.graph.graph.number_of_edges()

            if (
                current_nodes != state.ui_state.last_node_count
                or current_edges != state.ui_state.last_edge_count
            ):
                state.ui_state.last_node_count = current_nodes
                state.ui_state.last_edge_count = current_edges
                self._update_graph()
                self._update_sidebar()

        total = len(state.engine.round_snapshots) if state.engine.round_snapshots else 0

        if total > 0 and self.timeline_slider:
            self.timeline_slider.props(f"min=0 max={total - 1}")

            if not state.ui_state.replay_mode:
                self.timeline_slider.value = total - 1
                self.timeline_label.text = "实时模式"

    def _zoom_in(self):
        """Zoom in the graph."""
        if self.graph_chart:
            self.graph_chart.run_chart_method(
                "dispatchAction", {"type": "dataZoom", "zoom": 1.3}
            )

    def _zoom_out(self):
        """Zoom out the graph."""
        if self.graph_chart:
            self.graph_chart.run_chart_method(
                "dispatchAction", {"type": "dataZoom", "zoom": 0.7}
            )

    def _reset_graph(self):
        """Reset graph view to default."""
        if self.graph_chart:
            self.graph_chart.run_chart_method("dispatchAction", {"type": "restore"})
            self._update_graph()

    def _show_verdict(self):
        """Show the verdict dialog."""
        if state.engine.judgment_document:
            content = state.engine.judgment_document.replace("\n", "<br>")
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
        """Update all UI components."""
        state.ui_state.last_transcript_count = 0
        state.ui_state.last_node_count = 0
        state.ui_state.last_edge_count = 0
        self._update_graph()
        self._update_transcript()
        self._update_sidebar()
        self._update_button_states()

    def _update_graph(self):
        """Update the graph visualization."""
        if not self.graph_chart:
            return

        if not state.engine.graph:
            self._clear_graph()
            return

        try:
            option = EChartsAdapter.parse_graph(
                state.engine.graph, preferred_extension=state.engine.preferred_extension
            )

            self.graph_chart.run_chart_method("setOption", option, {"notMerge": True})

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
        """Update the transcript view."""
        if self.transcript_view:
            lines = state.engine.transcript or []
            self.transcript_view.refresh(lines)

        if self.transcript_count:
            count = len(state.engine.transcript) if state.engine.transcript else 0
            self.transcript_count.set_text(str(count))
            self.transcript_count.update()

    def _update_sidebar(self):
        """Update all sidebar cards."""
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


@ui.page("/")
def index():
    """Main page entry point."""
    app = LegalMASApp()
    app.build()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="Legal MAS Console",
        port=8080,
        reload=True,
        favicon="⚖️",
        dark=False,
    )
