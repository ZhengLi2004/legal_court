import pytest
import networkx as nx
from mas.common import ShadowGraph, NodeType, EdgeType, NodeStatus
from mas.backprop import BackPropagator

@pytest.fixture
def basic_graph():
    sg = ShadowGraph()
    sg.add_node("Fact A", NodeType.FACT, "agent_1")     # id: FACT_1
    sg.add_node("Claim B", NodeType.CLAIM, "agent_1")   # id: CLAIM_2
    sg.add_node("Claim C", NodeType.CLAIM, "agent_1")   # id: CLAIM_3
    sg.add_node("Fact D", NodeType.FACT, "agent_2")     # id: FACT_4
    sg.add_node("Claim E", NodeType.CLAIM, "agent_2")   # id: CLAIM_5
    return sg

def test_bpp1_anchoring_and_support_chain():
    sg = ShadowGraph()
    id_a = sg.add_node("Fact A", NodeType.FACT, "P")
    id_b = sg.add_node("Claim B", NodeType.CLAIM, "P")
    id_c = sg.add_node("Claim C", NodeType.CLAIM, "P")
    id_d = sg.add_node("Fact D", NodeType.FACT, "D")    # 无关点
    id_e = sg.add_node("Claim E", NodeType.CLAIM, "D")  # 攻击者
    sg.add_edge(id_a, id_b, EdgeType.SUPPORT)   # A -> B
    sg.add_edge(id_b, id_c, EdgeType.SUPPORT)   # B -> C
    sg.add_edge(id_e, id_c, EdgeType.CONFLICT)  # E -x-> C (攻击)
    bp = BackPropagator()
    bp.propagate(sg, explicit_validated_ids=[id_c])
    nodes = sg.graph.nodes
    assert nodes[id_c]['status'] == NodeStatus.VALIDATED, "Anchor C should be VALIDATED"
    assert nodes[id_b]['status'] == NodeStatus.VALIDATED, "Intermediate B should be VALIDATED via Support"
    assert nodes[id_a]['status'] == NodeStatus.VALIDATED, "Leaf Evidence A should be VALIDATED via Chain Support"
    assert nodes[id_d]['status'] == NodeStatus.HYPOTHETICAL, "Unrelated D should remain HYPOTHETICAL"
    assert nodes[id_e]['status'] == NodeStatus.HYPOTHETICAL, "Attacker E should not be affected in Step 1"

def test_bpp1_reset_logic():
    sg = ShadowGraph()
    id_a = sg.add_node("A", NodeType.FACT, "P")
    sg.graph.nodes[id_a]['status'] = NodeStatus.VALIDATED
    bp = BackPropagator()
    bp.propagate(sg, explicit_validated_ids=[])
    assert sg.graph.nodes[id_a]['status'] == NodeStatus.HYPOTHETICAL, "Node should be reset to HYPOTHETICAL"

def test_bpp2_conflict_kill_basic():
    sg = ShadowGraph()
    id_a = sg.add_node("Claim A", NodeType.CLAIM, "P")
    id_b = sg.add_node("Claim B", NodeType.CLAIM, "D")
    sg.add_edge(id_b, id_a, EdgeType.CONFLICT)  # B attacks A
    bp = BackPropagator()
    bp.propagate(sg, explicit_validated_ids=[id_a])
    nodes = sg.graph.nodes
    assert nodes[id_a]['status'] == NodeStatus.VALIDATED, "Anchor A should be Validated"
    assert nodes[id_b]['status'] == NodeStatus.DEFEATED, "Attacker B should be Defeated"

def test_bpp2_conflict_protection():
    sg = ShadowGraph()
    id_a = sg.add_node("Claim A", NodeType.CLAIM, "P")
    id_b = sg.add_node("Claim B", NodeType.CLAIM, "D")
    sg.add_edge(id_b, id_a, EdgeType.CONFLICT)
    bp = BackPropagator()
    bp.propagate(sg, explicit_validated_ids=[id_a, id_b])
    nodes = sg.graph.nodes
    assert nodes[id_a]['status'] == NodeStatus.VALIDATED
    assert nodes[id_b]['status'] == NodeStatus.VALIDATED, "Attacker B should remain Validated if explicitly anchored"

def test_bpp2_complex_topology():
    sg = ShadowGraph()
    id_f = sg.add_node("Fact F", NodeType.FACT, "P")
    id_a = sg.add_node("Claim A", NodeType.CLAIM, "P")
    id_b = sg.add_node("Claim B", NodeType.CLAIM, "D")
    sg.add_edge(id_f, id_a, EdgeType.SUPPORT)
    sg.add_edge(id_b, id_a, EdgeType.CONFLICT)
    bp = BackPropagator()
    bp.propagate(sg, explicit_validated_ids=[id_a])
    nodes = sg.graph.nodes
    assert nodes[id_a]['status'] == NodeStatus.VALIDATED
    assert nodes[id_f]['status'] == NodeStatus.VALIDATED, "Fact F should be Validated via Support"
    assert nodes[id_b]['status'] == NodeStatus.DEFEATED, "Claim B should be Defeated via Conflict"