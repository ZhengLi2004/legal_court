import pytest
from mas.common import ShadowGraph, ShadowNode, NodeType, ClaimMetadata, EdgeType
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
    projector._project_single_graph(tgt, src, [(tgt_fact_id, "New Fact")])
    found = False
    
    for nid, data in tgt.graph.nodes(data=True):
        if data['content'] == "Neighbor Claim":
            
            found = True
            assert "historical_status" in data['metadata']
            break
    
    assert found, "Neighbor node should be projected"

if __name__ == "__main__":
    pytest.main(["-v", "test/test_metadata.py"])