"""Main entry point for the Legal MAS Web GUI.

This module provides the entry point for running the NiceGUI application.
"""

from nicegui import ui

from apps.state import state
from apps.app import LegalMASApp


@ui.page("/")
def index():
    """Main page entry point."""
    app = LegalMASApp(state)
    app.build()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="Legal MAS Console",
        port=8080,
        reload=True,
        favicon="⚖️",
        dark=False,
    )