import pytest
import networkx as nx
from typing import List, Tuple
from mas.common import ShadowGraph, NodeType, EdgeType, NodeStatus
from mas.backprop import BackPropagator

class MockMatcher:
    def __init__(self, threshold=0.8): self.threshold = threshold

    def find_match(self, query: str, candidates: List[Tuple[str, str]]) -> str:
        query_key = query[:2]

        for cid, content in candidates:
            if content.startswith(query_key): return cid
        
        return None

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

def test_bpp3_basic_serialization():
    sg = ShadowGraph()
    sg.add_node("被告违约", NodeType.CLAIM, "P")
    text = sg.to_recursive_text()
    assert "[观点] 被告违约" in text

def test_bpp3_hierarchy():
    sg = ShadowGraph()
    id_f = sg.add_node("银行转账记录", NodeType.FACT, "P")
    id_c = sg.add_node("借贷关系成立", NodeType.CLAIM, "P")
    sg.add_edge(id_f, id_c, EdgeType.SUPPORT)
    text = sg.to_recursive_text()
    print(f"\n生成文本:\n{text}")
    assert "借贷关系成立" in text
    assert "支持依据" in text
    assert "银行转账记录" in text
    assert text.index("借贷关系成立") < text.index("银行转账记录")

def test_bpp3_conflict_branch():
    sg = ShadowGraph()
    id_a = sg.add_node("被告应赔偿", NodeType.CLAIM, "P")
    id_f = sg.add_node("损害鉴定书", NodeType.FACT, "P")
    id_b = sg.add_node("原告亦有过错", NodeType.CLAIM, "D")
    sg.add_edge(id_f, id_a, EdgeType.SUPPORT)
    sg.add_edge(id_b, id_a, EdgeType.CONFLICT)
    text = sg.to_recursive_text()
    print(f"\n生成文本:\n{text}")
    assert "支持依据" in text
    assert "损害鉴定书" in text
    assert "受到反驳" in text
    assert "原告亦有过错" in text

def test_bpp3_status_display():
    sg = ShadowGraph()
    id_win = sg.add_node("赢了的点", NodeType.CLAIM, "P")
    id_lose = sg.add_node("输了的点", NodeType.CLAIM, "D")
    sg.graph.nodes[id_win]['status'] = NodeStatus.VALIDATED
    sg.graph.nodes[id_lose]['status'] = NodeStatus.DEFEATED
    text = sg.to_recursive_text()
    assert "【已采信】" in text
    assert "【已驳回】" in text
    assert "赢了的点" in text

def test_bpp4_law_exact_match():
    sg = ShadowGraph()
    matcher = MockMatcher()
    id1 = sg.add_node("民法典第100条", NodeType.LAW, "P", matcher=matcher)
    id2 = sg.add_node("  民法典第100条 ", NodeType.LAW, "D", matcher=matcher)
    assert id1 == id2, "相同的法条应该被合并 ID"
    id3 = sg.add_node("民法典第101条", NodeType.LAW, "D", matcher=matcher)
    assert id1 != id3, "不同的法条(100 vs 101) 不应该合并，应忽略 Matcher"

def test_bpp4_fact_semantic_match():
    sg = ShadowGraph()
    matcher = MockMatcher()
    id1 = sg.add_node("张三打了李四", NodeType.FACT, "P", matcher=matcher)
    id2 = sg.add_node("张三殴打了李四", NodeType.FACT, "D", matcher=matcher)
    assert id1 == id2, "语义相似的事实应该被合并 (基于 Matcher)"

def test_bpp4_fallback_logic():
    sg = ShadowGraph()
    id1 = sg.add_node("事实A", NodeType.FACT, "P")
    id2 = sg.add_node("事实A", NodeType.FACT, "D")
    id3 = sg.add_node("事实B", NodeType.FACT, "D")
    assert id1 == id2
    assert id1 != id3