"""UI Components for the Legal MAS GUI.

This module provides all UI component classes for the application,
including views, cards, panels, and charts. Each component receives
state through dependency injection.
"""

from typing import Optional, Callable

from nicegui import ui

from apps.state import AppState


class TranscriptView:
    """A scrollable transcript view with auto-scroll capability."""

    def __init__(self, parent, state: AppState):
        """Initialize the TranscriptView component.

        Args:
            parent: The parent UI element.
            state: The application state instance.
        """
        self.state = state

        with parent:
            self.scroll_area = ui.scroll_area().classes(
                "w-full h-full bg-slate-50 rounded-lg"
            )

            with self.scroll_area:
                self.container = ui.column().classes("w-full gap-2 p-3")

        self._last_count = 0

    def refresh(self, lines: list[str]):
        """Update the transcript with new lines using incremental rendering.

        Performs a full re-render when the line count decreases (indicating
        a data source switch, e.g., entering replay mode), and incremental
        append when new lines are added.

        Args:
            lines: A list of transcript lines to display.
        """
        current_count = len(lines)

        if current_count < self._last_count:
            self.container.clear()
            self._last_count = 0

        if current_count == 0:
            if self._last_count == 0:
                self.container.clear()

                with self.container:
                    ui.label("等待辩论开始...").classes(
                        "text-gray-500 italic text-center py-8"
                    )

            self._last_count = 0
            return

        if current_count == self._last_count:
            return

        if self._last_count == 0 and current_count > 0:
            self.container.clear()

        with self.container:
            for i in range(self._last_count, current_count):
                self._render_line(lines[i], i)

        self._last_count = current_count
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

    def __init__(self, parent, state: AppState):
        """Initialize the StatusCard component.

        Args:
            parent: The parent UI element.
            state: The application state instance.
        """
        self.state = state

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

    def refresh(self):
        """Update the status card with current engine state."""
        self.round_label.text = str(self.state.engine.round_idx)
        turn = self.state.engine.current_turn

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

        if self.state.engine.convergence_history:
            conv = self.state.engine.convergence_history[-1]

        self.conv_label.text = f"{conv:.4f}"


class AgentStateCard:
    """A card displaying agent state machine status."""

    def __init__(self, parent, state: AppState):
        """Initialize the AgentStateCard component.

        Args:
            parent: The parent UI element.
            state: The application state instance.
        """
        self.state = state

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

    def _get_display_info(self, raw_state: str) -> tuple[str, str]:
        """Map raw backend state to display name and CSS color class.

        Handles shortening of long state names (e.g., ASSESS_NEEDS -> ASSESS).

        Args:
            raw_state: The raw state string from the backend engine.

        Returns:
            A tuple containing (display_name, css_classes).
        """
        name_map = {
            "IDLE": "IDLE",
            "ASSESS_NEEDS": "ASSESS",
            "WAIT_FOR_WORKERS": "WAIT",
            "DECIDE": "DECIDE",
            "DONE": "DONE",
        }

        display_name = name_map.get(raw_state, raw_state)

        colors = {
            "IDLE": "bg-gray-200 text-gray-700",
            "ASSESS": "bg-yellow-200 text-yellow-800",
            "WAIT": "bg-orange-200 text-orange-800",
            "DECIDE": "bg-indigo-200 text-indigo-800",
            "DONE": "bg-green-200 text-green-800",
        }

        css = colors.get(display_name, "bg-gray-200 text-gray-700")
        return display_name, css

    def refresh(self):
        """Update the agent state display."""
        self.content.clear()

        is_init = (
            self.state.engine.graph is not None
            and self.state.engine.p_team is not None
            and self.state.engine.d_team is not None
        )

        if not is_init:
            with self.content:
                self._render_placeholder()

            return

        p_raw = "IDLE"
        d_raw = "IDLE"

        try:
            if self.state.engine.p_team:
                p_raw = self.state.engine.p_team.pipeline_step.name

            if self.state.engine.d_team:
                d_raw = self.state.engine.d_team.pipeline_step.name

        except Exception as e:
            print(f"Error getting agent state: {e}")

        p_state, p_color = self._get_display_info(p_raw)
        d_state, d_color = self._get_display_info(d_raw)

        with self.content:
            with ui.row().classes("items-center gap-2 w-full"):
                ui.icon("person", color="blue").classes("text-lg")
                ui.label("原告:").classes("text-sm font-medium w-12")

                ui.label(p_state).classes(
                    f"px-2 py-1 rounded text-xs font-bold flex-1 text-center {p_color}"
                )

            with ui.row().classes("items-center gap-2 w-full"):
                ui.icon("person_outline", color="red").classes("text-lg")
                ui.label("被告:").classes("text-sm font-medium w-12")

                ui.label(d_state).classes(
                    f"px-2 py-1 rounded text-xs font-bold flex-1 text-center {d_color}"
                )

        self.content.update()


class StatsCard:
    """A card displaying graph statistics."""

    def __init__(self, parent, state: AppState):
        """Initialize the StatsCard component.

        Args:
            parent: The parent UI element.
            state: The application state instance.
        """
        self.state = state

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

        if not self.state.engine.graph:
            with self.content:
                self._render_empty()

            return

        try:
            G = self.state.engine.graph.graph

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

    def __init__(
        self, parent, state: AppState, on_view_full: Optional[Callable] = None
    ):
        """Initialize the JudgmentPreviewCard.

        Args:
            parent: The parent UI element.
            state: The application state instance.
            on_view_full: Callback function to show full verdict dialog.
        """
        self.state = state
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
        is_finished = self.state.engine.is_finished
        has_judgment = bool(self.state.engine.judgment_document)
        self.view_btn.visible = has_judgment
        self.view_btn.update()

        for _, indicator in self.phase_indicators.items():
            if is_finished:
                indicator.text = "✅"
            else:
                indicator.text = "⏳"


class NodeDetailsPanel:
    """A panel showing details of the selected node."""

    def __init__(self, parent, state: AppState):
        """Initialize the NodeDetailsPanel.

        Args:
            parent: The parent UI element.
            state: The application state instance.
        """
        self.state = state

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
        self.state.ui_state.show_node_details = False
        self.state.ui_state.selected_node_id = None
        self.card.visible = False
        self.card.update()

    def show_node(self, node_id: str):
        """Display details for the specified node.

        Args:
            node_id: The ID of the node to display.
        """
        if not self.state.engine.graph:
            return

        if not self.state.engine.graph.graph.has_node(node_id):
            return

        self.state.ui_state.selected_node_id = node_id
        self.state.ui_state.show_node_details = True
        self.card.visible = True
        node_data = self.state.engine.graph.graph.nodes[node_id]
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
            in_edges = list(self.state.engine.graph.graph.in_edges(node_id))
            out_edges = list(self.state.engine.graph.graph.out_edges(node_id))

            ui.label(f"连接: {len(in_edges)} 入 | {len(out_edges)} 出").classes(
                "text-sm text-gray-600"
            )

        self.card.update()


class ConvergenceChart:
    """A component for the convergence curve chart."""

    def __init__(self, parent, state: AppState):
        """Initialize the ConvergenceChart.

        Args:
            parent: The parent UI element.
            state: The application state instance.
        """
        self.state = state

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
        history = self.state.engine.convergence_history or []
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
