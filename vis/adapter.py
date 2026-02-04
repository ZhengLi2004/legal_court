"""Provides an adapter to convert the internal debate graph for ECharts.

This module defines the `EChartsAdapter`, a utility class that bridges the gap
between the application's internal data representation (the `ShadowGraph` based
on `networkx`) and the configuration object required by Apache ECharts.
"""

import textwrap
from typing import Any, Dict, Optional, Set, Tuple


class EChartsAdapter:
    """A static utility class to parse a debate graph into an ECharts option object.

    Attributes:
        COLORS: A dictionary mapping node types, statuses, and other properties
            to specific hex color codes for consistent styling.
    """

    COLORS = {
        "FACT": "#00CED1",
        "LAW": "#FFD700",
        "CLAIM_ROOT": "#2E8B57",
        "CLAIM_P": "#1E90FF",
        "CLAIM_D": "#DC143C",
        "CLAIM_COMMON": "#A9A9A9",
        "BORDER_VALID": "#32CD32",
        "BORDER_DEFEATED": "#8B0000",
        "BORDER_HYPO": "#696969",
        "EDGE_SUPPORT": "#32CD32",
        "EDGE_CONFLICT": "#FF4500",
    }

    @staticmethod
    def _extract_enum_value(value: Any) -> str:
        """Extract string value from enum or return string representation.

        Args:
            value: The value to extract from (may be enum or string).

        Returns:
            String representation of the value.
        """
        if value is None:
            return ""

        if hasattr(value, "value"):
            return str(value.value)

        if hasattr(value, "name"):
            return str(value.name)

        return str(value)

    @staticmethod
    def _get_category_name_and_color(node_data: Dict) -> Tuple[str, str, int, str]:
        """Determine the visual styling for a node based on its properties.

        Args:
            node_data: A dictionary of attributes for a single graph node.

        Returns:
            A tuple containing the category name, hex color code for fill,
            symbol size, and hex color code for border.
        """
        n_type = EChartsAdapter._extract_enum_value(node_data.get("type", "CLAIM"))
        n_type = n_type.upper()
        agent_id = str(node_data.get("agent_id", "")).lower()

        status = EChartsAdapter._extract_enum_value(
            node_data.get("status", "HYPOTHETICAL")
        )

        status = status.upper()
        color = EChartsAdapter.COLORS["CLAIM_COMMON"]
        category = "观点"
        symbol_size = 20

        if n_type == "FACT":
            color = EChartsAdapter.COLORS["FACT"]
            category = "事实"
            symbol_size = 18

        elif n_type == "LAW":
            color = EChartsAdapter.COLORS["LAW"]
            category = "法条"
            symbol_size = 18

        elif n_type == "CLAIM":
            symbol_size = 25

            if "system" in agent_id or "init" in agent_id:
                color = EChartsAdapter.COLORS["CLAIM_ROOT"]
                category = "核心诉求"
                symbol_size = 35

            elif "plaintiff" in agent_id:
                color = EChartsAdapter.COLORS["CLAIM_P"]
                category = "原告观点"

            elif "defendant" in agent_id:
                color = EChartsAdapter.COLORS["CLAIM_D"]
                category = "被告观点"

        border_color = EChartsAdapter.COLORS["BORDER_HYPO"]
        border_width = 0

        if status == "VALIDATED":
            border_color = EChartsAdapter.COLORS["BORDER_VALID"]
            border_width = 3

        elif status == "DEFEATED":
            border_color = EChartsAdapter.COLORS["BORDER_DEFEATED"]
            border_width = 3

        return category, color, symbol_size, border_color, border_width

    @staticmethod
    def parse_graph(
        graph_obj, preferred_extension: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """Convert a graph object into an ECharts configuration.

        Args:
            graph_obj: The graph object to parse. Can be a `ShadowGraph`
                instance or a raw `networkx` graph.
            preferred_extension: An optional set of node IDs that belong to the
                BAF preferred extension.

        Returns:
            A dictionary structured as a valid ECharts option object.
        """
        if graph_obj is None:
            return {"series": []}

        if hasattr(graph_obj, "graph"):
            G = graph_obj.graph

        else:
            G = graph_obj

        if G is None or not hasattr(G, "nodes"):
            return {"series": []}

        if preferred_extension is None:
            preferred_extension = set()

        nodes = []
        links = []
        category_color_map = {}

        for n_id, n_data in G.nodes(data=True):
            cat_name, color, size, border_color, border_width = (
                EChartsAdapter._get_category_name_and_color(n_data)
            )

            if cat_name not in category_color_map:
                category_color_map[cat_name] = color

            full_content = str(n_data.get("content", ""))
            wrapped_content = "<br/>".join(textwrap.wrap(full_content, width=40))

            item_style = {
                "color": color,
                "borderColor": border_color,
                "borderWidth": border_width,
            }

            if str(n_id) in preferred_extension:
                item_style["shadowBlur"] = 20
                item_style["shadowColor"] = "#FFD700"
                item_style["shadowOffsetX"] = 0
                item_style["shadowOffsetY"] = 0

            node_entry = {
                "id": str(n_id),
                "name": str(n_id),
                "value": wrapped_content,
                "symbolSize": size,
                "itemStyle": item_style,
                "category": cat_name,
                "label": {
                    "show": True,
                    "position": "bottom",
                    "fontSize": 10,
                    "color": "#333",
                    "formatter": "{b}",
                },
            }

            nodes.append(node_entry)

        for u, v, e_data in G.edges(data=True):
            e_type = EChartsAdapter._extract_enum_value(e_data.get("type", "SUPPORT"))
            e_type = e_type.upper()
            is_support = e_type == "SUPPORT"

            color = (
                EChartsAdapter.COLORS["EDGE_SUPPORT"]
                if is_support
                else EChartsAdapter.COLORS["EDGE_CONFLICT"]
            )

            links.append(
                {
                    "source": str(u),
                    "target": str(v),
                    "lineStyle": {
                        "color": color,
                        "width": 2,
                        "curveness": 0.15,
                        "type": "solid" if is_support else "dashed",
                    },
                    "symbol": ["none", "arrow"],
                    "symbolSize": [0, 8],
                }
            )

        categories_list = []

        for cat_name in sorted(category_color_map.keys()):
            categories_list.append(
                {"name": cat_name, "itemStyle": {"color": category_color_map[cat_name]}}
            )

        option = {
            "title": {
                "text": "辩论图谱",
                "subtext": f"节点: {len(nodes)} | 边: {len(links)}",
                "left": "center",
                "top": 5,
                "textStyle": {"fontSize": 14},
                "subtextStyle": {"fontSize": 10},
            },
            "tooltip": {
                "trigger": "item",
                "formatter": "{c}",
                "confine": True,
                "backgroundColor": "rgba(50,50,50,0.85)",
                "textStyle": {"color": "#fff", "fontSize": 12},
                "extraCssText": "max-width: 400px; white-space: normal;",
            },
            "legend": [
                {
                    "data": [{"name": c["name"]} for c in categories_list],
                    "orient": "horizontal",
                    "left": "center",
                    "bottom": 10,
                    "textStyle": {"fontSize": 11},
                }
            ],
            "animationDuration": 0,
            "animationDurationUpdate": 0,
            "series": [
                {
                    "type": "graph",
                    "layout": "force",
                    "data": nodes,
                    "links": links,
                    "categories": categories_list,
                    "roam": True,
                    "draggable": True,
                    "label": {
                        "show": True,
                        "position": "bottom",
                        "fontSize": 9,
                    },
                    "force": {
                        "repulsion": 350,
                        "edgeLength": [80, 180],
                        "gravity": 0.08,
                    },
                    "emphasis": {
                        "focus": "adjacency",
                        "lineStyle": {"width": 4},
                        "itemStyle": {"shadowBlur": 15},
                    },
                    "lineStyle": {
                        "opacity": 0.9,
                        "curveness": 0.15,
                    },
                }
            ],
        }

        return option
