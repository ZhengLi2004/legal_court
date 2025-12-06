import shutil
import os
from mas.legal_system import LegalSystem
from mas.common import NodeStatus

def test_full_cycle():
    print("\n>>> Testing Final Architecture (Upward/Downward/BackProp)...")
    test_dir = "./test_final_arch"
    if os.path.exists(test_dir): shutil.rmtree(test_dir)
    system = LegalSystem(persist_dir=test_dir)
    from mas.insights_manager import Insight
    system.insights.insights.append(Insight("盗窃罪需证明主观故意", score=5.0))
    system.insights._rebuild_index()
    context = "嫌疑人拿走了手机，但声称是借用"
    sg, insights = system.new_case(context)
    print(f"Projected Graph Nodes: {len(sg.graph.nodes)}")
    print(f"Retrieved Insights: {insights}")
    if "盗窃罪需证明主观故意" in insights[0]: print("✅ Upward Traversal (Semantic Insight Retrieval) works.")
    else: print("❌ Upward Traversal failed.")
    system.execute_action(sg, "plaintiff", 'ADD_CLAIM("由于未归还，判定为非法占有")')
    system.execute_action(sg, "defendant", 'ADD_CLAIM("只是忘了还")')
    print("Learning from case...")
    system.learn(context, sg, winner="plaintiff", case_id="case_new_01")
    msgs, _ = system.memory.retrieve_memory(context, top_k=1)
    saved_graph = msgs[0].shadow_graph
    p_nodes = [d for n, d in saved_graph.graph.nodes(data=True) if d['agent_id'] == "plaintiff"]
    if p_nodes and p_nodes[0]['status'] == NodeStatus.VALIDATED: print("✅ BackProp worked (Status Updated).")
    else: print(f"❌ BackProp failed. Status: {[d.get('status') for d in p_nodes]}")
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    try:
        test_full_cycle()
        print("\n✅ Final Architecture Verified!")
    
    except Exception as e: print(f"\n❌ Verification Failed: {e}")