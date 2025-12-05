import shutil
import os
from mas.insights_manager import InsightsManager
from mas.judge import LLMJudge
from mas.llm import GPTChat
from mas.common import ShadowGraph, NodeType

def test_adversarial_insight_and_judge():
    print("\n>>> Testing Adversarial Insight & Judge (Step 11+)...")
    test_dir = "./test_insights_adv"
    if os.path.exists(test_dir): shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    llm = GPTChat()
    print("Testing Judge...")
    judge = LLMJudge(llm)
    sg = ShadowGraph()
    sg.add_node("被告人当场被抓获，赃物在身", NodeType.FACT, "plaintiff")
    is_settled, winner = judge.evaluate("盗窃案", sg)
    print(f"Judge Verdict: Settled={is_settled}, Winner={winner}")
    print("Testing Insight Extraction...")
    manager = InsightsManager(test_dir, llm)
    win_g = ShadowGraph()
    win_g.add_node("被告人有主观非法占有目的", NodeType.CLAIM, "plaintiff")
    lose_g = ShadowGraph()
    lose_g.add_node("被告人只是想借用", NodeType.CLAIM, "defendant")
    
    manager.extract_adversarial_insights(
        case_id="case_001",
        case_context="盗窃案，被告人辩称是借用。",
        winning_graph=win_g,
        losing_graph=lose_g
    )
    
    if len(manager.insights) > 0: print(f"✅ Adversarial Insight: {manager.insights[0].content}")
    
    else:
        print("❌ Failed to extract insight.")
        raise AssertionError("Insight extract failed")

    shutil.rmtree(test_dir)

if __name__ == "__main__":
    try:
        test_adversarial_insight_and_judge()
        print("\n✅ Step 9 Completed Successfully!")
    
    except Exception as e: print(f"\n❌ Step 9 Failed: {e}")