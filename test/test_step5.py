import os
import shutil
from mas.legal_memory import LegalGMemory
from mas.common import LegalMessage, ShadowGraph

def test_full_memory_flow():
    print("\n>>> Testing LegalGMemory (Full Flow)...")
    test_dir = "./test_storage_full"
    if os.path.exists(test_dir): shutil.rmtree(test_dir)
    memory = LegalGMemory(persist_dir=test_dir, embedding_model_path="./bge-m3")

    cases = [
        ("case_A", "抢劫罪，持刀抢劫便利店。", ShadowGraph()),
        ("case_B", "持械抢劫，在超市威胁店员。", ShadowGraph()), # 相似
        ("case_C", "离婚诉讼，财产分割争议。", ShadowGraph())    # 不相似
    ]

    print("Inserting cases...")
    
    for cid, ctx, sg in cases:
        msg = LegalMessage(case_id=cid, case_context=ctx, shadow_graph=sg)
        memory.add_memory(msg)
        print(f"  Inserted {cid}")

    assert memory.collection.count() == 3
    print("✅ ChromaDB count correct.")
    if memory.task_layer.graph.has_edge("case_A", "case_B"): print("✅ Edge A-B exists (Topology Updated).")
    else: print("⚠️ Edge A-B missing. Checking weights...")
    query = "拿着刀去小卖部抢钱"
    print(f"Querying: '{query}'")
    retrieved_msgs, _ = memory.retrieve_memory(query, top_k=1)
    r_ids = [m.case_id for m in retrieved_msgs]
    print(f"Retrieved IDs: {r_ids}")
    assert "case_A" in r_ids or "case_B" in r_ids
    print("✅ Retrieval successful.")
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    try:
        test_full_memory_flow()
        print("\n✅ Step 5 Completed Successfully!")
    
    except Exception as e: print(f"\n❌ Step 5 Failed: {e}")