"""Panel components for the Legal MAS GUI.

This module provides panel-style UI components, such as node details panels.
"""

from nicegui import ui

from apps.state import AppState


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
