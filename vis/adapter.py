"""Provides an adapter to convert the internal debate graph for ECharts.

This module defines the `EChartsAdapter`, a utility class that bridges the gap
between the application's internal data representation (the `ShadowGraph` based
on `networkx`) and the configuration object required by the Apache ECharts
charting library. It handles the translation of nodes, edges, and their
properties (like type, status, and agent ownership) into visual attributes such
as colors, sizes, and styles.
"""

import textwrap
from typing import Any, Dict, Tuple


class EChartsAdapter:
    """A static utility class to parse a debate graph into an ECharts option object.

    This class provides static methods to handle the conversion. It contains a
    predefined color scheme for different graph elements and logic to map node
    and edge attributes to specific visual properties, creating a rich,
    interactive, and informative graph visualization for the frontend.

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
    }

    @staticmethod
    def _get_category_name_and_color(node_data: Dict) -> Tuple[str, str, int, str]:
        """Determine the visual styling for a node based on its properties.

        This helper method inspects a node's 'type', 'agent_id', and 'status'
        attributes to assign it a legend category, a fill color, a symbol size,
        and a border color. This logic centralizes the visual styling rules for
        the graph.

        Args:
            node_data: A dictionary of attributes for a single graph node.

        Returns:
            A tuple containing the category name (e.g., "Fact", "Plaintiff Claim"),
            hex color code for the node's fill, an integer for the symbol size,
            and a hex color code for the node's border.
        """
        n_type = node_data.get("type", "CLAIM")

        if hasattr(n_type, "value"):
            n_type = n_type.value

        n_type = str(n_type)
        agent_id = node_data.get("agent_id", "").lower()
        status = node_data.get("status", "HYPOTHETICAL")

        if hasattr(status, "value"):
            status = status.value

        color = EChartsAdapter.COLORS["CLAIM_COMMON"]
        category = "观点"
        symbol_size = 20

        if n_type == "FACT":
            color = EChartsAdapter.COLORS["FACT"]
            category = "事实"
            symbol_size = 15

        elif n_type == "LAW":
            color = EChartsAdapter.COLORS["LAW"]
            category = "法条"
            symbol_size = 15

        elif n_type == "CLAIM":
            symbol_size = 25

            if "system" in agent_id or "init" in agent_id:
                color = EChartsAdapter.COLORS["CLAIM_ROOT"]
                category = "核心诉求"
                symbol_size = 30

            elif "plaintiff" in agent_id:
                color = EChartsAdapter.COLORS["CLAIM_P"]
                category = "原告观点"

            elif "defendant" in agent_id:
                color = EChartsAdapter.COLORS["CLAIM_D"]
                category = "被告观点"

        border_color = EChartsAdapter.COLORS["BORDER_HYPO"]

        if status == "VALIDATED":
            border_color = EChartsAdapter.COLORS["BORDER_VALID"]

        elif status == "DEFEATED":
            border_color = EChartsAdapter.COLORS["BORDER_DEFEATED"]

        return category, color, symbol_size, border_color

    @staticmethod
    def parse_graph(graph_obj) -> Dict[str, Any]:
        """Convert a graph object into an ECharts configuration.

        This method iterates over the nodes and edges of the input graph,
        translating each element into the corresponding structure required by
        ECharts. It builds the `nodes`, `links`, and `categories` lists and
        assembles them into the final ECharts option dictionary.

        Args:
            graph_obj: The graph object to parse. Can be a `ShadowGraph`
                instance or a raw `networkx` graph.

        Returns:
            A dictionary structured as a valid ECharts option object, ready to
            be consumed by the frontend.
        """
        if hasattr(graph_obj, "graph"):
            G = graph_obj.graph

        else:
            G = graph_obj

        nodes = []
        links = []
        category_color_map = {}

        for n_id, n_data in G.nodes(data=True):
            cat_name, color, size, border_color = (
                EChartsAdapter._get_category_name_and_color(n_data)
            )

            if cat_name not in category_color_map:
                category_color_map[cat_name] = color

            full_content = n_data.get("content", "")
            wrapped_content = "<br/>".join(textwrap.wrap(full_content, width=40))

            item_style = {
                "color": color,
                "borderColor": border_color,
                "borderWidth": 3
                if border_color != EChartsAdapter.COLORS["BORDER_HYPO"]
                else 0,
            }

            nodes.append(
                {
                    "id": str(n_id),
                    "name": str(n_id),
                    "value": wrapped_content,
                    "symbolSize": size,
                    "itemStyle": item_style,
                    "category": cat_name,
                    "label": {"show": False},
                }
            )

        for u, v, e_data in G.edges(data=True):
            e_type = e_data.get("type", "SUPPORT")

            if hasattr(e_type, "value"):
                e_type = e_type.value

            is_support = str(e_type) == "SUPPORT"
            color = "#32CD32" if is_support else "#FF4500"

            links.append(
                {
                    "source": str(u),
                    "target": str(v),
                    "lineStyle": {
                        "color": color,
                        "width": 2,
                        "curveness": 0.1,
                        "type": "solid" if is_support else "dashed",
                    },
                }
            )

        categories_list = []

        for cat_name in sorted(category_color_map.keys()):
            categories_list.append(
                {"name": cat_name, "itemStyle": {"color": category_color_map[cat_name]}}
            )

        option = {
            "title": {"text": "Debate Graph", "bottom": 0, "right": 0},
            "tooltip": {
                "trigger": "item",
                "formatter": "{c}",
                "confine": True,
                "backgroundColor": "rgba(50,50,50,0.7)",
                "textStyle": {"color": "#fff", "fontSize": 12},
            },
            "legend": [
                {
                    "data": categories_list,
                    "orient": "horizontal",
                    "left": "center",
                    "top": "top",
                }
            ],
            "series": [
                {
                    "type": "graph",
                    "layout": "force",
                    "data": nodes,
                    "links": links,
                    "categories": categories_list,
                    "roam": True,
                    "label": {"show": False},
                    "force": {
                        "repulsion": 300,
                        "edgeLength": [50, 150],
                        "gravity": 0.1,
                    },
                }
            ],
        }

        return option
