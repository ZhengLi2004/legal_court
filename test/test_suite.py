import os
import shutil
import threading
import time
import pytest
from mas.llm import GPTChat, Message
from mas.utils import EmbeddingFunc, cosine_similarity, simple_file_lock
from mas.common import ShadowGraph, NodeType, EdgeType, NodeStatus, LegalMessage
from mas.graph_ops import GraphExecutor
from mas.task_layer import TaskLayer
from mas.legal_memory import LegalGMemory
from mas.semantic_matcher import SemanticMatcher
from mas.projection import GraphProjector
from mas.insights_manager import InsightsManager
from mas.judge import LLMJudge
from mas.legal_system import LegalSystem
from vis.recorder import SystemRecorder
from vis.dynamic_viz import generate_dynamic_gif
from vis.visualize_memory_state import snapshot_global_state
TEST_DIR = "./demo"

@pytest.fixture(scope="function")
def cleanup():
    if os.path.exists(TEST_DIR): shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    yield
    if os.path.exists(TEST_DIR): shutil.rmtree(TEST_DIR)
# Phase 1: Infrastructure Tests
def test_p1_infrastructure(cleanup):
    print("\n--- Testing Phase 1: Infrastructure ---")
    llm = GPTChat()
    res = llm([Message(role="user", content="你好")])
    assert len(res) > 0, "LLM call failed"
    print("✅ LLM works.")
    ef = EmbeddingFunc(model_path="./bge-m3")
    sim = cosine_similarity(ef.embed_query("盗窃"), ef.embed_query("偷窃"))
    assert sim > 0.8, "Embedding similarity calculation is off"
    print("✅ Embedding works.")
# Phase 2: Core Component Tests
def test_p2_core_components(cleanup):
    print("\n--- Testing Phase 2: Core Components ---")
    ef = EmbeddingFunc(model_path="./bge-m3")
    matcher = SemanticMatcher(ef, threshold=0.8)
    sg = ShadowGraph()
    executor = GraphExecutor(sg, matcher)
    logs = executor.execute_batch('ADD_FACT("偷了钱包"); ADD_LAW("刑法")', "plaintiff")
    assert sg.graph.number_of_nodes() == 2
    logs = executor.execute_batch('LINK(FACT_1, LAW_1, SUPPORT)', "plaintiff")
    assert sg.graph.number_of_edges() == 1
    print("✅ ShadowGraph & Executor work.")
    id1 = sg.add_node("盗窃", NodeType.FACT, "a", matcher)
    id2 = sg.add_node("偷窃", NodeType.FACT, "b", matcher)
    assert id1 == id2, "Semantic Matcher failed to merge"
    print("✅ Semantic Matcher works.")
# Phase 3: Memory & Projection Tests
def test_p3_memory_and_projection(cleanup):
    print("\n--- Testing Phase 3: Memory & Projection ---")
    system = LegalSystem(persist_dir=TEST_DIR)
    seed_context = "被告人张三深夜撬锁进入李四家中，盗窃了财物。"
    seed_sg = ShadowGraph()
    
    system.execute_action(seed_sg, "plaintiff", """
        ADD_FACT("被告人实施了撬锁入户")
        ADD_LAW("刑法264条 盗窃罪")
        ADD_CLAIM("入户盗窃属于加重情节")
        LINK(FACT_1, LAW_1, SUPPORT)
        LINK(FACT_1, CLAIM_1, SUPPORT)
    """)

    system.learn(seed_context, seed_sg, "plaintiff", "seed_01")
    new_context = "本案中，被告人深夜撬锁进入了被害人家中。"
    sg_new, _ = system.new_case(new_context)
    system.execute_action(sg_new, "plaintiff", 'ADD_FACT("被告人实施了撬锁入户")')
    projected = [d for n, d in sg_new.graph.nodes(data=True) if d.get('agent_id') == 'projection']
    print(f"Projected Nodes ({len(projected)}):")
    for p_node in projected: print(f"  - {p_node['type']}: {p_node['content']}")
    assert len(projected) >= 2, "Projection should import at least LAW and CLAIM"
    projected_contents = {p['content'] for p in projected}
    assert "刑法264条 盗窃罪" in projected_contents
    assert "入户盗窃属于加重情节" in projected_contents
    print("✅ Projection works as expected.")
    assert system.memory.task_layer.graph.has_node("seed_01")
    print("✅ TaskLayer is being populated.")
# Phase 4: Full System & Visualization Demo
def test_p4_full_demo(cleanup):
    print("\n--- Testing Phase 4: Full Demo & Visualization (Enhanced Trilogy)...")
    root_dir = TEST_DIR
    storage_dir = os.path.join(root_dir, "storage")
    trace_file = os.path.join(root_dir, "trace.json")
    gif_file = os.path.join(root_dir, "demo.gif")
    recorder = SystemRecorder(trace_file)
    system = LegalSystem(persist_dir=storage_dir, recorder=recorder)
    print("\n--- Running Seed Case 1: Contract Fraud ---")
    seed_context_1 = "被告人李四伪造公司资质，与原告签订大额采购合同，收款后失联。"
    seed_sg_1 = ShadowGraph()
    
    system.execute_action(seed_sg_1, "plaintiff", """
        ADD_FACT("伪造公司资质签订合同")
        ADD_LAW("刑法224条 合同诈骗罪")
        ADD_CLAIM("具有非法占有目的")
        LINK(FACT_1, LAW_1, SUPPORT)
        LINK(FACT_1, CLAIM_1, SUPPORT)
    """)
    
    system.learn(seed_context_1, seed_sg_1, "plaintiff", "case_seed_01")
    print("\n--- Running Reinforcement Case 2: Supply Fraud ---")
    seed_context_2 = "被告人赵五虚构货源，与原告签订供货协议，收取定金后无法交货。"
    seed_sg_2 = ShadowGraph()

    system.execute_action(seed_sg_2, "plaintiff", """
        ADD_FACT("虚构货源签订协议")
        ADD_LAW("刑法224条 合同诈骗罪")
        LINK(FACT_1, LAW_1, SUPPORT)
    """)

    system.learn(seed_context_2, seed_sg_2, "plaintiff", "case_seed_02")
    print("\n--- Running Ultimate Test Case 3: Fundraising Fraud ---")
    context = "被告人王五编造元宇宙养猪项目，向公众非法集资后挥霍。"
    sg_live, _ = system.new_case(context)
    print("  > Turn 1: Plaintiff acts...")
    p_action_1 = 'ADD_FACT("编造虚假项目骗取款项")'
    system.execute_action(sg_live, "plaintiff", p_action_1)
    print("  > Turn 2: Defendant acts...")
    d_action_1 = 'ADD_CLAIM("这是正常的商业投资失败")'
    system.execute_action(sg_live, "defendant", d_action_1)
    print("  > Turn 3: Plaintiff acts...")
    p_action_2 = 'ADD_FACT("被告人将资金用于个人奢侈品消费")'
    system.execute_action(sg_live, "plaintiff", p_action_2)
    print("  > Turn 4: Plaintiff links projected nodes...")
    p_action_3 = 'LINK(FACT_1, LAW_2, SUPPORT)' 
    system.execute_action(sg_live, "plaintiff", p_action_3)
    system.learn(context, sg_live, "plaintiff", "case_test_hybrid")
    recorder.save()
    generate_dynamic_gif(trace_file, gif_file, duration=2.0)
    snapshot_global_state(system, 3, output_dir=os.path.join(root_dir, "viz_output"))
    assert os.path.exists(gif_file), "GIF generation failed"
    print(f"\n✅ Trilogy Demo with Visualization Completed. See {gif_file}")

def run_all_tests():
    test_p1_infrastructure(cleanup())
    test_p2_core_components(cleanup())
    test_p3_memory_and_projection(cleanup())
    test_p4_full_demo(cleanup())

if __name__ == "__main__": run_all_tests()