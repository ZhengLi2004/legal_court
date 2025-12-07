import pytest
from mas.common import ShadowGraph, LegalMessage, NodeStatus, ShadowNode, NodeType, ClaimMetadata, EdgeType
from mas.projection import GraphProjector
from mas.config import SystemConfig

# Mock classes
class MockEmbeddingFunc:
    def embed_query(self, text): return [0.1, 0.2]

class MockMatcher:
    def __init__(self): self.embedding_func = MockEmbeddingFunc()
    def find_match(self, q, candidates): return candidates[0][0] if candidates else None

def test_metadata_structure():
    meta: ClaimMetadata = {
        "is_root_claim": True,
        "claim_index": 1,
        "created_at": 123456.789
    }
    
    node = ShadowNode(
        id="CLAIM_1",
        content="Test Claim",
        type=NodeType.CLAIM,
        agent_id="Agent_A",
        metadata=meta
    )
    
    assert node.metadata["is_root_claim"] is True

def test_metadata_compatibility():
    node = ShadowNode(
        id="FACT_1",
        content="Test Fact",
        type=NodeType.FACT,
        agent_id="Agent_B",
        metadata={"random_field": "some_value"}
    )

    assert node.metadata["random_field"] == "some_value"

def test_projection_metadata_structure():
    src = ShadowGraph()
    src_fact_id = src.add_node("Old Fact", NodeType.FACT, "Agent_Old")
    tgt = ShadowGraph()
    tgt_fact_id = tgt.add_node("New Fact", NodeType.FACT, "System") 
    matcher = MockMatcher()
    projector = GraphProjector(matcher, SystemConfig())
    src_claim_id = src.add_node("Neighbor Claim", NodeType.CLAIM, "Agent_Old")
    src.add_edge(src_fact_id, src_claim_id, EdgeType.SUPPORT)
    projector._project_single_graph(tgt, src, [(tgt_fact_id, "New Fact")], case_id="TEST_CASE_ID")
    found = False
    
    for nid, data in tgt.graph.nodes(data=True):
        if data['content'] == "Neighbor Claim":
            
            found = True
            assert "historical_status" in data['metadata']
            break
    
    assert found, "Neighbor node should be projected"

def test_status_metadata_projection():
    src_sg = ShadowGraph()
    anchor_id = src_sg.add_node("Anchor Fact", NodeType.FACT, "P")
    bad_claim_id = src_sg.add_node("Bad Claim", NodeType.CLAIM, "P")
    src_sg.graph.nodes[bad_claim_id]['status'] = NodeStatus.DEFEATED
    src_sg.add_edge(anchor_id, bad_claim_id, EdgeType.SUPPORT)
    msg = LegalMessage(case_id="CASE_2024", case_context="ctx", shadow_graph=src_sg)
    tgt_sg = ShadowGraph()
    tgt_anchor = tgt_sg.add_node("Anchor Fact", NodeType.FACT, "Sys")
    projector = GraphProjector(MockMatcher(), SystemConfig())
    projector.project(tgt_sg, [msg])
    found_bad = False

    for _, data in tgt_sg.graph.nodes(data=True):
        if data['content'] == "Bad Claim":
            found_bad = True
            assert data['metadata']['projected_from_case'] == "CASE_2024"
            assert data['metadata']['historical_status'] == "DEFEATED"

    assert found_bad, "Defeated node should be projected (Full History)"

def test_historical_status_display():
    sg = ShadowGraph()
    meta = {"historical_status": "DEFEATED", "projected_from_case": "OLD_CASE"}
    id_bad = sg.add_node("高利贷利息", NodeType.CLAIM, "Sys", metadata=meta)
    text = sg.to_recursive_text()
    print(f"\n历史状态文本:\n{text}")
    assert "高利贷利息" in text
    assert "历史教训" in text
    assert "曾被驳回" in text

def test_historical_success_display():
    sg = ShadowGraph()
    meta = {"historical_status": NodeStatus.VALIDATED}
    id_good = sg.add_node("合法利息", NodeType.CLAIM, "Sys", metadata=meta)
    text = sg.to_recursive_text()
    assert "历史经验" in text
    assert "曾被采信" in text

if __name__ == "__main__":
    pytest.main(["-v", "test/test_metadata.py"])