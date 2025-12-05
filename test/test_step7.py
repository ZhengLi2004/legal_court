from mas.common import ShadowGraph, NodeType, EdgeType
from mas.semantic_matcher import SemanticMatcher
from mas.projection import GraphProjector
from mas.utils import EmbeddingFunc


def test_associative_projection():
    print("\n>>> Testing Associative Projection (Step 9)...")
    ef = EmbeddingFunc(model_path="./bge-m3")
    matcher = SemanticMatcher(embedding_func=ef, threshold=0.90)
    projector = GraphProjector(matcher)
    hist_graph = ShadowGraph()
    h_fact = hist_graph.add_node("嫌疑人实施盗窃", NodeType.FACT, "agent_h")
    h_law = hist_graph.add_node("刑法第264条", NodeType.LAW, "agent_h")
    hist_graph.add_edge(h_fact, h_law, EdgeType.SUPPORT)
    print("History Graph: Fact -> Law 264")
    curr_graph = ShadowGraph()
    c_fact = curr_graph.add_node("嫌疑人偷东西", NodeType.FACT, "agent_c")
    print(f"Current Graph Init: Node {c_fact}")
    assert curr_graph.graph.number_of_nodes() == 1
    print("Projecting history to current...")
    projector.project(curr_graph, [hist_graph])
    nodes = list(curr_graph.graph.nodes(data=True))
    print(f"Current Graph Nodes after Projection: {len(nodes)}")
    
    if len(nodes) != 2:
        print(f"❌ Nodes count mismatch. Got {len(nodes)}, expected 2.")
        raise AssertionError("Projection failed to add node.")
        
    has_law = False
    for nid, data in nodes:
        if data['type'] == NodeType.LAW and "264" in data['content']: has_law = True
            
    if not has_law:
        print("❌ Projected Law node missing.")
        raise AssertionError("Law node missing")
        
    if curr_graph.graph.number_of_edges() == 1: print("✅ Edge successfully projected from Synonym Fact to Law.")
    
    else:
        print("❌ Edge missing.")
        raise AssertionError("Edge projection failed")

if __name__ == "__main__":
    try:
        test_associative_projection()
        print("\n✅ Step 7 Completed Successfully!")
    
    except Exception as e: print(f"\n❌ Step 7 Failed: {e}")