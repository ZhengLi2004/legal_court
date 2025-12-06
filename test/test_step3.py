from mas.common import ShadowGraph
from mas.graph_ops import GraphExecutor

def test_executor():
    print("\n>>> Testing Graph Executor...")
    sg = ShadowGraph()
    executor = GraphExecutor(sg)

    llm_output = """
    根据案情，我建议执行以下操作：
    1. ADD_FACT("嫌疑人携带管制刀具")
    2. ADD_LAW("刑法第264条")
    3. LINK(FACT_1, LAW_2, SUPPORT)
    """

    logs = executor.execute_batch(llm_output)
    print("Execution Logs:", logs)
    nodes = list(sg.graph.nodes(data=True))
    edges = list(sg.graph.edges(data=True))
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {len(edges)}")
    assert len(nodes) == 2
    assert len(edges) == 1
    assert nodes[0][1]['content'] == "嫌疑人携带管制刀具"
    assert edges[0][2]['type'] == 'SUPPORT'
# Step 3: 测试图操作
if __name__ == "__main__":
    try:
        test_executor()
        print("\n✅ Step 3 Completed Successfully!")
    
    except Exception as e: print(f"\n❌ Step 4 Failed: {e}")