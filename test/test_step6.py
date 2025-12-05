from mas.common import ShadowGraph, NodeType
from mas.semantic_matcher import SemanticMatcher
from mas.utils import EmbeddingFunc, cosine_similarity

def test_semantic_deduplication():
    print("\n>>> Testing Semantic Deduplication (Step 8)...")
    ef = EmbeddingFunc(model_path="./bge-m3")
    THRESHOLD = 0.90
    matcher = SemanticMatcher(embedding_func=ef, threshold=THRESHOLD)
    sg = ShadowGraph()
    vec_base = ef.embed_query("嫌疑人实施了盗窃行为")
    vec_syn = ef.embed_query("嫌疑人进行了偷窃")
    vec_diff = ef.embed_query("嫌疑人实施了抢劫行为")
    sim_syn = cosine_similarity(vec_base, vec_syn)
    sim_diff = cosine_similarity(vec_base, vec_diff)
    print(f"Similarity (盗窃 vs 偷窃): {sim_syn:.4f}")
    print(f"Similarity (盗窃 vs 抢劫): {sim_diff:.4f}")
    if sim_syn < THRESHOLD: print("⚠️ Warning: Synonym similarity is lower than threshold!")
    if sim_diff >= THRESHOLD: print("⚠️ Warning: Distinct similarity is higher than threshold! Threshold needs adjustment.")
    id1 = sg.add_node("嫌疑人实施了盗窃行为", NodeType.FACT, "agent_1", matcher=matcher)
    print(f"Added Base Node: {id1}")
    id2 = sg.add_node("嫌疑人进行了偷窃", NodeType.FACT, "agent_2", matcher=matcher)
    print(f"Added Synonym Node: {id2}")
    if id1 == id2: print("✅ Semantic Deduplication Worked (Synonym merged).")
    else: print(f"❌ Failed to merge synonym. IDs: {id1} vs {id2}")
    id3 = sg.add_node("嫌疑人实施了抢劫行为", NodeType.FACT, "agent_3", matcher=matcher)
    print(f"Added Distinct Node: {id3}")
    if id1 != id3: print("✅ Distinct concept kept separate.")
    
    else:
        print("❌ Incorrectly merged distinct concepts.")
        raise AssertionError("Distinct merge error")
    
    id4 = sg.add_node("嫌疑人实施了盗窃行为", NodeType.CLAIM, "agent_4", matcher=matcher)
    print(f"Added Same Content Different Type: {id4}")
    if id1 != id4: print("✅ Type isolation worked.")
    
    else:
        print("❌ Incorrectly merged different types.")
        raise AssertionError("Type isolation error")
    
if __name__ == "__main__":
    try:
        test_semantic_deduplication()
        print("\n✅ Step 6 Completed Successfully!")
    
    except Exception as e: print(f"\n❌ Step 6 Failed: {e}")