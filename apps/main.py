"""Main entry point for the Legal MAS Web GUI.

This module provides the entry point for running the NiceGUI application.
"""

from nicegui import ui

from apps.app import LegalMASApp
from apps.state import create_app_state


@ui.page("/")
def index():
    """Main page entry point.

    Creates an independent AppState per browser session to prevent
    cross-user state interference.
    """
    session_state = create_app_state()
    app = LegalMASApp(session_state)
    app.build()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="Legal MAS Console",
        port=8080,
        reload=False,
        favicon="⚖️",
        dark=False,
    )
