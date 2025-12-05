import os
import shutil
from mas.task_layer import TaskLayer
from mas.utils import EmbeddingFunc

def test_query_graph():
    print("\n>>> Testing TaskLayer (Query Graph)...")
    test_dir = "./test_storage"
    if os.path.exists(test_dir): shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    ef = EmbeddingFunc(model_path="./bge-m3")
    task_layer = TaskLayer(working_dir=test_dir, embedding_func=ef, similarity_threshold=0.6)

    cases = [
        ("case_1", "被告人入室盗窃，偷得现金5000元。"),
        ("case_2", "嫌疑人潜入居民家中窃取财物。"),  # 应该与 case_1 相似
        ("case_3", "双方因合同纠纷发生争执。")       # 应该不相似
    ]

    for cid, context in cases:
        print(f"Adding {cid}...")
        task_layer.add_case_node(context, cid)

    if task_layer.graph.has_edge("case_1", "case_2"): print("✅ Edge created between case_1 and case_2 (Similar).")
    
    else:
        print("❌ Failed to create edge for similar cases.")
        raise AssertionError("Edge missing")
    
    if not task_layer.graph.has_edge("case_1", "case_3"): print("✅ No edge between case_1 and case_3 (Dissimilar).")
    
    else:
        print("❌ Incorrect edge created for dissimilar cases.")
        raise AssertionError("Incorrect edge")
    # 测试 Hop 检索
    query = "小偷破门而入盗走金项链。"
    results = task_layer.retrieve_related_cases(query, top_k=1, hop=1)
    print(f"Query Results: {results}")
    assert "case_1" in results or "case_2" in results
    if "case_1" in results and "case_2" in results: print("✅ Hop retrieval worked: Neighbors included.")
    shutil.rmtree(test_dir)
# Step 4: 测试 Query Graph
if __name__ == "__main__":
    try:
        test_query_graph()
        print("\n✅ Step 4 Completed Successfully!")
    
    except Exception as e: print(f"\n❌ Step 4 Failed: {e}")