import os
import shutil
from typing import List
from mas.common import ShadowGraph, NodeType, NodeStatus, EdgeType, LegalMessage
from mas.legal_system import LegalSystem
from vis.recorder import SystemRecorder
from vis.dynamic_viz import generate_dynamic_gif

DEMO_DIR = "./demo"

def setup_env():
    if os.path.exists(DEMO_DIR): shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)
    print(f"✅ Environment cleaned and setup at: {DEMO_DIR}")

def print_section(title): print(f"\n{'='*20} {title} {'='*20}")

def find_node_by_content(graph: ShadowGraph, substring: str) -> str:
    for n, d in graph.graph.nodes(data=True):
        if substring in d['content']: return n

    return None

def execute_turn(system: LegalSystem, graph: ShadowGraph, agent: str, action: str, step_desc: str):
    print(f"  - {step_desc}")
    
    if hasattr(system, 'recorder') and system.recorder:
        graph.id_alias.clear()

        system.recorder.log_event(
            step_name=f"Turn: {agent}",
            shadow_graph=graph,
            message=f"{step_desc}\nCmd: {action[:50]}..."
        )

    logs = system.execute_action(graph, agent, action)
    return logs

def prepare_historical_memory(system: LegalSystem) -> LegalSystem:
    print_section("Module 1: Preparing Historical Memory")
    case_a_ctx = "被告复制了原告的核心算法代码，代码MD5比对完全一致。"
    sg_a = ShadowGraph()
    f_a = sg_a.add_node("代码MD5比-对完全一致", NodeType.FACT, "plaintiff")
    c_a = sg_a.add_node("侵犯软件著作权", NodeType.CLAIM, "plaintiff")
    sg_a.add_edge(f_a, c_a, EdgeType.SUPPORT)
    sg_a.graph.nodes[f_a]['status'] = NodeStatus.VALIDATED
    sg_a.graph.nodes[c_a]['status'] = NodeStatus.VALIDATED
    msg_a = LegalMessage(case_id="CASE_WIN", case_context=case_a_ctx, shadow_graph=sg_a)
    system.memory.add_memory(msg_a)
    print("  - Stored [Win] Precedent: Code Copyright (with VALIDATED status)")
    case_b_ctx = "被告使用了客户名单，但客户信息在公网可查。"
    sg_b = ShadowGraph()
    c_b = sg_b.add_node("侵犯商业秘密", NodeType.CLAIM, "plaintiff")
    f_b = sg_b.add_node("客户信息在公网可查", NodeType.FACT, "defendant")
    c_attack = sg_b.add_node("信息不具秘密性", NodeType.CLAIM, "defendant")
    sg_b.add_edge(f_b, c_attack, EdgeType.SUPPORT)
    sg_b.add_edge(c_attack, c_b, EdgeType.CONFLICT)
    sg_b.graph.nodes[c_b]['status'] = NodeStatus.DEFEATED       # 原告的诉求被驳回
    sg_b.graph.nodes[c_attack]['status'] = NodeStatus.VALIDATED # 被告的反驳观点成立
    sg_b.graph.nodes[f_b]['status'] = NodeStatus.VALIDATED      # 被告的证据被采信
    msg_b = LegalMessage(case_id="CASE_LOSE", case_context=case_b_ctx, shadow_graph=sg_b)
    system.memory.add_memory(msg_b)
    print("  - Stored [Lose] Precedent: Trade Secret (with DEFEATED/VALIDATED status)")
    assert system.memory.collection.count() == 2
    print("✅ Verification successful: Memory contains 2 precedents with correct statuses.")
    return system

def demonstrate_retrieval(system: LegalMessage) -> List[LegalMessage]:
    print_section("Module 2.A: Demonstrating Retrieval")
    query_context = "本案涉及代码复制和客户名单公开问题。"
    print(f"  Querying with context: '{query_context}'")
    retrieved_messages, _ = system.memory.retrieve_memory(query_context, top_k=2)
    assert len(retrieved_messages) >= 2, f"Expected to retrieve 2 cases, but got {len(retrieved_messages)}."
    print("  ✅ Retrieved correct number of historical cases.")
    retrieved_ids = {msg.case_id for msg in retrieved_messages}
    assert "CASE_WIN" in retrieved_ids, "Failed to retrieve the [Win] case."
    assert "CASE_LOSE" in retrieved_ids, "Failed to retrieve the [Lose] case."
    print("  ✅ Correct historical cases (CASE_WIN, CASE_LOSE) were retrieved.")
    return retrieved_messages

def demonstrate_projection(system: LegalSystem, retrieved_messages: List[LegalMessage]) -> ShadowGraph:
    print_section("Module 2.B: Demonstrating Projection")
    sg_live = ShadowGraph()
    sg_live.add_node("代码MD5比对完全一致", NodeType.FACT, "plaintiff")
    sg_live.add_node("客户信息在公网可查", NodeType.FACT, "defendant")
    system.projector.project(sg_live, retrieved_messages)
    proj_win_node = find_node_by_content(sg_live, "侵犯软件著作权")
    proj_lose_neighbor = find_node_by_content(sg_live, "信息不具秘密性")
    assert proj_win_node is not None
    assert proj_lose_neighbor is not None, "Failed to project direct neighbor from Case B."
    win_meta = sg_live.graph.nodes[proj_win_node].get('metadata', {})
    lose_neighbor_meta = sg_live.graph.nodes[proj_lose_neighbor].get('metadata', {})
    assert win_meta.get('historical_status') == "VALIDATED"
    assert lose_neighbor_meta.get('historical_status') == "VALIDATED"
    print("  ✅ Projected direct neighbors correctly carry their historical status.")
    return sg_live

def demonstrate_battle(system: LegalSystem, graph: ShadowGraph) -> ShadowGraph:
    print_section("Module 3.A: Demonstrating Semantic Merge")
    system.dedup_matcher.threshold = 0.80
    print(f"  - Set deduplication threshold to {system.dedup_matcher.threshold}")
    execute_turn(system, graph, "plaintiff", 'ADD_CLAIM("被告窃取了核心客户资料")', "原告提出商业秘密主张")
    p_secret_id = find_node_by_content(graph, "窃取")
    nodes_before = graph.graph.number_of_nodes()
    exact_fact_text = "客户信息在公网能查"
    action_d1 = f'ADD_FACT("{exact_fact_text}")'
    execute_turn(system, graph, "defendant", action_d1, "被告提出与历史证据相同的抗辩")
    nodes_after = graph.graph.number_of_nodes()
    assert nodes_after == nodes_before, "Merge failed!"
    print("  ✅ [Highlight] Semantic Merge successful!")
    merged_node_id = find_node_by_content(graph, "客户信息在公网可查")
    assert merged_node_id is not None
    
    execute_turn(system, graph, "defendant", 
                 f'ADD_CLAIM("信息不具秘密性"); CHALLENGE(CLAIM_1, {p_secret_id}, {merged_node_id})', 
                 "被告基于合并后的节点发起攻击")

    print_section("Module 3.B: Activation & Integrity")
    execute_turn(system, graph, "plaintiff", 'ADD_CLAIM("被告代码构成实质性相似")', "原告提出代码侵权主张")
    p_code_id = find_node_by_content(graph, "实质性相似")
    proj_win_node = find_node_by_content(graph, "代码MD5比对完全一致")
    assert proj_win_node is not None, "Pre-condition failed: Winning projection not found."
    
    execute_turn(system, graph, "plaintiff", 
                 f"SUPPORT({proj_win_node}, {p_code_id})", 
                 "激活历史胜诉经验")

    print("\n  - Defendant counters on the code issue...")
    mit_fact_text = "涉案代码在GitHub上以MIT协议开源"
    mit_claim_text = "原告主张的代码侵权不成立，因代码已开源"
    
    execute_turn(system, graph, "defendant", 
                 f'ADD_FACT("{mit_fact_text}"); ADD_CLAIM("{mit_claim_text}"); CHALLENGE(CLAIM_1, {p_code_id}, FACT_1)',
                 "被告提出 MIT 开源抗辩")
        
    print("  ✅ [Highlight] Projection Activation successful!")
    print("\n  Verifying Integrity Checks...")
    test_c_id = find_node_by_content(graph, "实质性相似")
    test_f_id = find_node_by_content(graph, "客户信息在公网可查")
    logs_a = execute_turn(system, graph, "defendant", f"SUPPORT({test_c_id}, {test_f_id})", "尝试非法拓扑连接")
    assert any("Invalid connection" in log for log in logs_a), "Failed to block C->F."
    print("  ✅ Integrity check [1/2] passed.")
    logs_b = execute_turn(system, graph, "defendant", f"CHALLENGE({test_c_id}, {p_code_id})", "尝试无证据攻击")
    assert any("No evidence cited" in log for log in logs_b), "Failed to block baseless CHALLENGE."
    print("  ✅ Integrity check [2/2] passed.")
    return graph

def demonstrate_judge_and_learning(system: LegalSystem, graph: ShadowGraph):
    print_section("Module 4: Judge & Learning")
    d_secret_rebuttal_id = find_node_by_content(graph, "信息不具秘密性")
    p_code_claim_id = find_node_by_content(graph, "实质性相似")
    d_mit_rebuttal_id = find_node_by_content(graph, "代码已开源")
    p_secret_claim_id = find_node_by_content(graph, "窃取了核心客户资料")
    assert d_mit_rebuttal_id is not None
    assert p_secret_claim_id is not None
    winning_ids = [d_mit_rebuttal_id, p_secret_claim_id]
    print(f"  - Judge anchors on: {winning_ids}")
    final_graph = system.backprop.propagate(graph, winning_ids)
    print("  - Back-propagation completed.")
    s_mit = final_graph.graph.nodes[d_mit_rebuttal_id]['status']
    s_code = final_graph.graph.nodes[p_code_claim_id]['status']
    s_secret = final_graph.graph.nodes[p_secret_claim_id]['status']
    s_secret_rebuttal = final_graph.graph.nodes[d_secret_rebuttal_id]['status']
    print("\n  Verifying final graph statuses...")
    assert s_mit == NodeStatus.VALIDATED, "Defendant's MIT rebuttal should be VALIDATED."
    assert s_code == NodeStatus.DEFEATED, "Plaintiff's code claim should be DEFEATED by the MIT rebuttal."
    print("  ✅ [Highlight] Code issue verdict is correct (Offensive Kill successful).")
    assert s_secret == NodeStatus.VALIDATED, "Plaintiff's secret claim should be VALIDATED."
    assert s_secret_rebuttal == NodeStatus.DEFEATED, "Defendant's rebuttal on secret should be DEFEATED."
    print("  ✅ [Highlight] Trade secret issue verdict is correct (Defensive Kill successful).")
    return final_graph

if __name__ == "__main__":
    setup_env()
    trace_file = os.path.join(DEMO_DIR, "trace.json")
    recorder = SystemRecorder(trace_file)
    system = LegalSystem(persist_dir=DEMO_DIR, recorder=recorder)
    system_with_memory = prepare_historical_memory(system)
    retrieved = demonstrate_retrieval(system_with_memory)
    live_graph = demonstrate_projection(system_with_memory, retrieved)
    graph_after_battle = demonstrate_battle(system_with_memory, live_graph)
    final_adjudicated_graph = demonstrate_judge_and_learning(system, graph_after_battle)
    recorder.log_event("Final Verdict", final_adjudicated_graph, "The final state of the argumentation graph.")
    recorder.save()
    print_section("Epilogue: Generating Visualization")
    gif_path = os.path.join(DEMO_DIR, "demo_showcase.gif")
    generate_dynamic_gif(trace_file, gif_path, duration=2.5)
    print("\n--- End of Modules ---")