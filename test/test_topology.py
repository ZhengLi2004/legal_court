import pytest
from mas.common import ShadowGraph, NodeType
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

if __name__ == "__main__": pytest.main(["-v", "test/test_topology.py"])