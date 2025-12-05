import shutil
import os
from mas.legal_system import LegalSystem
from mas.common import ShadowGraph, NodeType

def test_legal_system_facade():
    print("\n>>> Testing LegalSystem Facade (Pre-MetaGPT Check)...")
    test_dir = "./test_system_storage"
    if os.path.exists(test_dir): shutil.rmtree(test_dir)
    system = LegalSystem(persist_dir=test_dir)
    print("Seeding memory...")
    hist_g = ShadowGraph()
    hist_g.add_node("被告人盗窃", "FACT", "plaintiff", matcher=system.matcher)
    hist_g.add_node("刑法264", "LAW", "plaintiff", matcher=system.matcher)
    hist_g.add_edge("FACT_1", "LAW_2", "SUPPORT")
    
    system.learn(
        context="历史盗窃案", 
        win_graph=hist_g, 
        lose_graph=ShadowGraph(),
        case_id="case_hist_01"
    )

    print("Starting new case...")
    context = "被告人盗窃"
    current_graph = system.new_case(context)
    nodes = list(current_graph.graph.nodes(data=True))
    print(f"Projected Graph Nodes: {len(nodes)}")
    if len(nodes) >= 2: print("✅ Projection via System worked.")
    else: print("❌ Projection failed.")
    print("Executing Agent Action...")
    logs = system.execute_action(current_graph, "plaintiff", 'ADD_CLAIM("应当定罪")')
    print(f"Action Logs: {logs}")
    print("Adjudicating...")
    settled, winner = system.adjudicate(context, current_graph)
    print(f"Verdict: {settled}, {winner}")
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    try:
        test_legal_system_facade()
        print("\n✅ System Facade Check Completed!")
    
    except Exception as e: print(f"\n❌ System Check Failed: {e}")