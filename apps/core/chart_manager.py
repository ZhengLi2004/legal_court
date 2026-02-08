"""Chart manager for the Legal MAS GUI.

This module provides chart management logic for all chart operations.
"""

from nicegui import ui

from apps.core.app_orchestrator import AppOrchestrator


class ChartManager:
    """Chart manager that handles all chart operations.

    This class is responsible for resizing charts and animating chart
    resize operations. It receives an AppOrchestrator instance to access
    state and components.
    """

    def __init__(self, orchestrator: AppOrchestrator):
        """Initialize the chart manager.

        Args:
            orchestrator: The application orchestrator instance.
        """
        self.orchestrator = orchestrator

    def _resize_charts(self) -> None:
        """Resize all ECharts instances once.

        This method is used for non-animated layout changes or post-render
        corrections (e.g., after setOption). It keeps the implementation
        centralized so callers do not need to know which charts exist.
        """
        if self.orchestrator.graph_chart:
            self.orchestrator.graph_chart.run_chart_method("resize")

        if self.orchestrator.convergence_chart and getattr(
            self.orchestrator.convergence_chart, "chart", None
        ):
            self.orchestrator.convergence_chart.chart.run_chart_method("resize")

    def _animate_chart_resize(
        self,
        duration: float = 0.28,
        interval: float = 0.05,
    ) -> None:
        """Continuously resize charts during a CSS width transition.

        Side panels expand/collapse with a CSS width transition. During this time,
        the center area width changes continuously. ECharts does not automatically
        follow container size changes, so the canvas can appear to lag or overlap
        with adjacent panels.

        This method starts a short-lived timer to call `resize()` repeatedly during
        the transition and performs one final resize at the end.

        Args:
            duration: Total seconds to keep resizing. Should be slightly larger
                than the CSS transition duration (e.g., 0.25s -> 0.28s).
            interval: Resize cadence in seconds (e.g., 0.05s).
        """
        prev_timer = getattr(self.orchestrator, "_chart_resize_timer", None)

        if prev_timer:
            prev_timer.cancel()

        prev_stop = getattr(self.orchestrator, "_chart_resize_stop_timer", None)

        if prev_stop:
            prev_stop.cancel()

        self._resize_charts()
        self.orchestrator._chart_resize_timer = ui.timer(interval, self._resize_charts)

        def _stop_and_finalize() -> None:
            timer = getattr(self.orchestrator, "_chart_resize_timer", None)

            if timer:
                timer.cancel()
                self.orchestrator._chart_resize_timer = None

            self._resize_charts()

        self.orchestrator._chart_resize_stop_timer = ui.timer(
            duration, _stop_and_finalize, once=True
        )
