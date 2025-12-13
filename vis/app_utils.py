import streamlit as st
from mas.engine import DebateEngine
from mas.config import SystemConfig
from .adapter import GraphAdapter
import os

JUDGE_CONFIG = {
    "model_name": "法衡",
    "temperature": 0.0,
    "max_tokens": 512
}

def initialize_engine():
    if 'engine' not in st.session_state:
        st.session_state.engine = DebateEngine(
            config=SystemConfig(),
            judge_config=JUDGE_CONFIG
        )
    
    return st.session_state.engine

def render_graph(graph_obj):
    from streamlit_agraph import agraph, Config
    from mas.common import ShadowGraph
    graph_dict_data = ShadowGraph.to_dict(graph_obj)
    graph_dict = graph_dict_data["graph_data"]
    nodes, edges = GraphAdapter.parse_shadow_graph(graph_dict)
    
    config = Config(
        width=800,
        height=600,
        directed=True, 
        physics=True, 
        nodeHighlightBehavior=True,
    )

    agraph(nodes=nodes, edges=edges, config=config)

def render_global_memory(snapshot):
    from .static_viz import GMemoryVisualizer
    viz = GMemoryVisualizer(output_dir="./temp_viz")
    im = snapshot.get("insight_manager")
    
    if im and im.insights:
        viz.draw_insight_graph(im, filename="insights.png")
        if os.path.exists("./temp_viz/insights.png"): st.image("./temp_viz/insights.png", caption="Insight Graph")
    
    else: st.info("💡 暂无已学习的 Insight。")

    tl = snapshot.get("task_layer")
    
    if tl and tl.graph.number_of_nodes() > 0:
        viz.draw_query_graph(tl, filename="topology.png")
        if os.path.exists("./temp_viz/topology.png"): st.image("./temp_viz/topology.png", caption="Case Topology Graph")
    
    else: st.info("🕸️ 暂无案件拓扑关联。")