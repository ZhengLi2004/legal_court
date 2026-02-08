"""Chart components for the Legal MAS GUI.

This module provides chart-style UI components, such as convergence charts.
"""

from nicegui import ui

from apps.state import AppState


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
