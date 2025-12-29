from streamlit_agraph import Node, Edge
from networkx.readwrite import json_graph

class GraphAdapter:
    COLORS = {
        "FACT": "#87CEFA",      # 浅蓝
        "LAW": "#FFD700",       # 金色
        "CLAIM": "#90EE90",     # 浅绿
        "BORDER_VALID": "green",
        "BORDER_DEFEATED": "red",
        "BORDER_HYPO": "gray"
    }

    @staticmethod
    def get_node_config(node_data: dict, node_id: str):
        n_type = node_data.get('type', 'CLAIM')
        if hasattr(n_type, 'value'): n_type = n_type.value
        n_type = str(n_type)
        status = node_data.get('status', 'HYPOTHETICAL')
        if hasattr(status, 'value'): status = status.value
        status = str(status)
        content = node_data.get('content', '')
        label = f"[{n_type}]\n{content[:10]}..." if len(content) > 10 else f"[{n_type}]\n{content}"
        meta = node_data.get('metadata', {})
        tooltip = f"ID: {node_id}\nType: {n_type}\nStatus: {status}\n\nContent:\n{content}\n\nMeta: {meta}"

        return Node(
            id=node_id,
            label=label,
            size=25,
            color=GraphAdapter.COLORS.get(n_type, "gray"),
            title=tooltip,
            borderWidth=3,
            borderColor=GraphAdapter.COLORS.get(f"BORDER_{status}", "gray"),
            shape="dot"
        )
    
    @staticmethod
    def get_edge_config(source, target, edge_data):
        e_type = edge_data.get('type', 'SUPPORT')
        if hasattr(e_type, 'value'): e_type = e_type.value
        e_type = str(e_type)
        color = "green" if e_type == "SUPPORT" else "red"

        return Edge(
            source=source,
            target=target,
            label=e_type,
            color=color,
            type="CURVE_SMOOTH"
        )
    
    @staticmethod
    def parse_shadow_graph(graph_dict: dict):
        nodes = []
        edges = []
        G = json_graph.node_link_graph(graph_dict)
        for n_id, n_data in G.nodes(data=True): nodes.append(GraphAdapter.get_node_config(n_data, n_id))
        for u, v, e_data in G.edges(data=True): edges.append(GraphAdapter.get_edge_config(u, v, e_data))
        return nodes, edges