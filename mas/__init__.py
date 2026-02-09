"""A multi-agent system (MAS) for simulating legal debates.

This package provides the core components for a multi-agent system designed
to simulate legal arguments.
"""

from .core.engine import DebateEngine
from .core.graph import ShadowGraph
from .core.schemas import AgentAction
from .core.system import LegalSystem

__all__ = ["DebateEngine", "LegalSystem", "ShadowGraph", "AgentAction"]
