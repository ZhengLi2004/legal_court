from mas.common import ShadowGraph, NodeType
from mas.graph_ops import GraphExecutor
from mas.semantic_matcher import SemanticMatcher
from mas.utils import EmbeddingFunc

def test_executor_upgrade():
    print("\n>>> Testing Graph Executor Upgrade (Step 10)...")
    
    ef = EmbeddingFunc(model_path="./bge-m3")
    matcher = SemanticMatcher(embedding_func=ef, threshold=0.90)
    sg = ShadowGraph()
    executor = GraphExecutor(sg, matcher)
    
    # 1. Plaintiff 行动
    output_p = """
    我认为：
    1. ADD_FACT("被告人偷了钱包")
    """
    logs = executor.execute_batch(output_p, agent_id="plaintiff")
    print(f"Plaintiff Logs: {logs}")
    
    # 获取节点 ID
    nodes = list(sg.graph.nodes(data=True))
    id_1 = nodes[0][0]
    owner_1 = nodes[0][1]['agent_id']
    print(f"Node 1 Owner: {owner_1}")
    assert owner_1 == "plaintiff"

    # 2. Defendant 行动 (尝试添加语义相同的节点)
    output_d = """
    反驳：
    1. ADD_FACT("被告人窃取了钱包")  <-- 语义相同
    2. ADD_CLAIM("这是误会")
    """
    logs = executor.execute_batch(output_d, agent_id="defendant")
    print(f"Defendant Logs: {logs}")
    
    # 验证去重
    # 如果去重成功，图中应该仍然只有 1 个 FACT 节点，且 owner 保持为 plaintiff (先入为主)
    # 或者逻辑是：如果已存在，add_node 返回旧ID，不修改 owner。我们看 common.py 实现。
    # common.py: if existing_id: return existing_id. -> 不修改 owner。
    
    fact_nodes = [n for n, d in sg.graph.nodes(data=True) if d['type'] == NodeType.FACT]
    print(f"Fact Nodes Count: {len(fact_nodes)}")
    
    if len(fact_nodes) == 1:
        print("✅ Semantic Deduplication via Executor worked.")
    else:
        print("❌ Deduplication failed.")
        raise AssertionError("Dedup failed")

    # 验证第二个节点 (CLAIM) 是否归属 Defendant
    claim_nodes = [d for n, d in sg.graph.nodes(data=True) if d['type'] == NodeType.CLAIM]
    if claim_nodes[0]['agent_id'] == "defendant":
        print("✅ Agent attribution worked for new node.")
    else:
        print("❌ Agent attribution error.")
        raise AssertionError("Attribution failed")

if __name__ == "__main__":
    try:
        test_executor_upgrade()
        print("\n✅ Step 10 Completed Successfully!")
    except Exception as e:
        print(f"\n❌ Step 10 Failed: {e}")