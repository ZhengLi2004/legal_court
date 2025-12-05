from mas.common import ShadowGraph, NodeType, EdgeType, LegalMessage

def test_shadow_graph_ops():
    print("\n>>> Testing Shadow Graph Operations...")
    sg = ShadowGraph()
    fact_id = sg.add_node("被告人进入房间", NodeType.FACT, "agent_fact")
    law_id = sg.add_node("刑法264条", NodeType.LAW, "agent_law")
    print(f"Created Nodes: {fact_id}, {law_id}")
    assert fact_id.startswith("FACT_")
    assert law_id.startswith("LAW_")
    dup_id = sg.add_node("被告人进入房间", NodeType.FACT, "agent_fact_2")
    print(f"Duplicate Node ID: {dup_id} (Should be {fact_id})")
    assert dup_id == fact_id
    sg.add_edge(fact_id, law_id, EdgeType.SUPPORT)
    assert sg.graph.has_edge(fact_id, law_id)
    print("Edge added successfully.")
    msg = LegalMessage(case_id="case_001", case_context="测试案件", shadow_graph=sg)
    json_data = LegalMessage.to_dict(msg)
    restored_msg = LegalMessage.from_dict(json_data)
    restored_graph = restored_msg.shadow_graph.graph
    print(f"Restored Graph Nodes: {restored_graph.nodes(data=True)}")
    assert len(restored_graph.nodes) == 2
    assert len(restored_graph.edges) == 1
# Step 2: 定义 Shadow Graph 数据结构
if __name__ == "__main__":
    try:
        test_shadow_graph_ops()
        print("\n✅ Step 2 Completed Successfully!")
    
    except Exception as e: print(f"\n❌ Step 2 Failed: {e}")