"""Card components for the Legal MAS GUI.

This module provides card-style UI components, including status cards,
agent state cards, statistics cards, and judgment preview cards.
"""

from typing import Callable, Optional

from nicegui import ui

from apps.state import AppState


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
