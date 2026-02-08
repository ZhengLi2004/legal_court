"""State management for the Legal MAS GUI.

This module provides centralized state management classes for the application,
including execution state, UI state, and global application state.
"""

from dataclasses import dataclass, field
from typing import Optional

from data.loader import CaseDataLoader
from mas.config import SystemConfig
from mas.engine import DebateEngine


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