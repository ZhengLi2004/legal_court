"""State management for the Legal MAS GUI.

This module provides centralized state management classes for the application,
including execution state, UI state, and global application state.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from data.loader import CaseDataLoader
from mas.config import SystemConfig
from mas.core.engine import DebateEngine


@dataclass
class UIState:
    """Centralized UI state management."""

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
    left_sidebar_visible: bool = True
    right_panel_visible: bool = True
    view_revision: int = 0


@dataclass
class AppState:
    """Manage the global state of the web application.

    Each browser session should create its own instance to prevent
    cross-user state corruption.

    Attributes:
        engine: The debate engine instance.
        loader: The case data loader.
        samples: List of loaded case samples.
        selected_index: Index of the currently selected case.
        ui_state: The UI state instance.
        _lock: Asyncio lock to prevent concurrent engine operations.
    """

    engine: DebateEngine = field(
        default_factory=lambda: DebateEngine(config=SystemConfig(), judge_config={})
    )

    loader: CaseDataLoader = field(
        default_factory=lambda: CaseDataLoader("data/sampling")
    )

    samples: list = field(default_factory=list)
    selected_index: int = 0
    ui_state: UIState = field(default_factory=UIState)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        """Load samples after initialization.

        Raises:
            RuntimeError: If no samples can be loaded and the application
                cannot function without case data.
        """
        try:
            self.samples = self.loader.load_all(limit=20)

        except Exception as e:
            print(f"Warning: Failed to load samples: {e}")
            self.samples = []

        if not self.samples:
            print(
                "Warning: No case samples loaded. The application may not function correctly."
            )

    @property
    def current_case(self):
        """Get the currently selected CaseData object.

        Returns:
            The current CaseData object, or None if the index is invalid
            or no samples are loaded.
        """
        if self.samples and 0 <= self.selected_index < len(self.samples):
            return self.samples[self.selected_index]

        return None

    @property
    def is_initialized(self) -> bool:
        """Check if the engine has been initialized.

        Returns:
            True if the engine graph has been created.
        """
        return self.engine.graph is not None

    def set_selected_index(self, value: int) -> None:
        """Set the selected case index with bounds checking.

        Args:
            value: The index to set. Will be clamped to valid range.
        """
        if not self.samples:
            self.selected_index = 0
            return

        self.selected_index = max(0, min(value, len(self.samples) - 1))

    def reset(self):
        """Reset the engine, UI state, and selected index."""
        self.engine = DebateEngine(config=SystemConfig(), judge_config={})
        self.ui_state = UIState()
        self.selected_index = 0


def create_app_state() -> AppState:
    """Create a new AppState instance for a browser session.

    Returns:
        A fresh AppState instance with loaded samples.
    """
    return AppState()
