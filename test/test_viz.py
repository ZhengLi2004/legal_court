from mas.common import ShadowGraph, NodeType, EdgeType, NodeStatus
from vis.recorder import SystemRecorder
from vis.dynamic_viz import generate_dynamic_gif

def test_visualization_pipeline():
    print("\n>>> Testing Visualization Pipeline...")
    trace_file = "viz_test_trace.json"
    gif_file = "viz_test_output.gif"
    recorder = SystemRecorder(trace_file)
    sg = ShadowGraph()
    sg.add_node("案情：盗窃嫌疑", NodeType.FACT, "system")
    recorder.log_event("Init", sg, "Case Started")
    nid_f = sg.add_node("事实：拿走钱包", NodeType.FACT, "plaintiff")
    nid_l = sg.add_node("法律：刑法264", NodeType.LAW, "plaintiff")
    sg.add_edge(nid_f, nid_l, EdgeType.SUPPORT)
    recorder.log_event("Plaintiff", sg, "Established Evidence Chain")
    nid_c = sg.add_node("抗辩：借用", NodeType.CLAIM, "defendant")
    sg.add_edge(nid_c, nid_f, EdgeType.CONFLICT)
    recorder.log_event("Defendant", sg, "Challenged Fact")
    sg.graph.nodes[nid_f]['status'] = NodeStatus.VALIDATED
    sg.graph.nodes[nid_l]['status'] = NodeStatus.VALIDATED
    sg.graph.nodes[nid_c]['status'] = NodeStatus.DEFEATED
    recorder.log_event("Verdict", sg, "Plaintiff Wins (Fact Validated)")
    recorder.save()
    generate_dynamic_gif(trace_file, gif_file, duration=1.0)

if __name__ == "__main__": test_visualization_pipeline()