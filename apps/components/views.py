"""View components for the Legal MAS GUI.

This module provides view-specific UI components, such as the transcript view.
"""

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
