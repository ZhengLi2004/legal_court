import pytest
from mas.common import ShadowGraph, NodeType, EdgeType
from mas.graph_ops import GraphExecutor

def test_api_simplification():
    sg = ShadowGraph()
    ex = GraphExecutor(sg)
    c1 = sg.add_node("Claim 1", NodeType.CLAIM, "P")
    c2 = sg.add_node("Claim 2", NodeType.CLAIM, "P")
    f1 = sg.add_node("Fact 1", NodeType.FACT, "P")
    sg.id_alias["C1"] = c1
    sg.id_alias["C2"] = c2
    sg.id_alias["F1"] = f1
    logs = ex.execute_batch("SUPPORT(F1, C1)", "P")
    assert any("Support Added" in log for log in logs)
    assert sg.graph.has_edge(f1, c1)
    logs = ex.execute_batch("CHALLENGE(C2, C1, F1)", "D")
    assert any("Challenge Added" in log for log in logs)
    assert sg.graph.has_edge(f1, c2) # 自动补全 Support
    assert sg.graph.has_edge(c2, c1) # 自动建立 Conflict

def test_topology_constraints_via_support():
    sg = ShadowGraph()
    ex = GraphExecutor(sg)
    c = sg.add_node("Claim", NodeType.CLAIM, "P")
    f = sg.add_node("Fact", NodeType.FACT, "P")
    sg.id_alias["C"] = c
    sg.id_alias["F"] = f
    logs = ex.execute_batch("SUPPORT(C, F)", "P")
    assert any("Invalid connection" in log for log in logs)

def test_challenge_constraints():
    sg = ShadowGraph()
    ex = GraphExecutor(sg)
    tgt = sg.add_node("Tgt", NodeType.CLAIM, "P")
    att = sg.add_node("Att", NodeType.CLAIM, "D")
    bad_evi = sg.add_node("Bad", NodeType.CLAIM, "D") # 观点不可作证据
    sg.id_alias["T"] = tgt
    sg.id_alias["A"] = att
    sg.id_alias["B"] = bad_evi
    logs = ex.execute_batch("CHALLENGE(A, T)", "D")
    assert any("No evidence" in log for log in logs)
    logs = ex.execute_batch("CHALLENGE(A, T, B)", "D")
    assert any("not valid" in log for log in logs)

def test_tactical_view_filtering():
    sg = ShadowGraph()
    r1 = sg.add_node("Root 1", NodeType.CLAIM, "P", metadata={"is_root_claim": True})
    e1 = sg.add_node("E1", NodeType.FACT, "P")
    c1 = sg.add_node("C1", NodeType.CLAIM, "P")
    c2 = sg.add_node("C2", NodeType.CLAIM, "P")
    sg.add_edge(e1, c1, EdgeType.SUPPORT)
    sg.add_edge(c1, c2, EdgeType.SUPPORT)
    sg.add_edge(c2, r1, EdgeType.SUPPORT)
    a1 = sg.add_node("A1", NodeType.CLAIM, "D")
    sg.add_edge(a1, c2, EdgeType.CONFLICT)
    r2 = sg.add_node("Root 2", NodeType.CLAIM, "P", metadata={"is_root_claim": True})
    e2 = sg.add_node("E2", NodeType.FACT, "P")
    c3 = sg.add_node("C3", NodeType.CLAIM, "P")
    sg.add_edge(e2, c3, EdgeType.SUPPORT)
    sg.add_edge(c3, r2, EdgeType.SUPPORT)
    c_future = sg.add_node("C_Future", NodeType.CLAIM, "P")
    sg.add_edge(c2, c_future, EdgeType.SUPPORT)
    sub_sg = sg.get_tactical_subgraph(focus_nodes=[c2])
    nodes = sub_sg.graph.nodes
    print(f"\n保留的节点: {[sub_sg.graph.nodes[n]['content'] for n in nodes]}")
    assert c2 in nodes, "Focus node C2 must be kept"
    assert c1 in nodes, "Parent C1 must be kept"
    assert e1 in nodes, "Grandparent E1 must be kept"
    assert a1 in nodes, "Attacker A1 (Predecessor) must be kept"
    assert r1 in nodes, "Successor R1 must be kept"
    assert c_future in nodes, "Successor C_Future must be kept"
    assert r2 in nodes, "Inactive Root R2 must be kept as anchor"
    assert c3 not in nodes, "Parallel branch C3 should be pruned"
    assert e2 not in nodes, "Parallel branch E2 should be pruned"

if __name__ == "__main__": pytest.main(["-v", "test/test_topology.py"])