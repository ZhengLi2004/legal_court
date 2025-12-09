import os
import shutil
from mas.common import ShadowGraph, NodeType, NodeStatus, EdgeType
from mas.legal_system import LegalSystem

try:
    from vis.recorder import SystemRecorder
    from vis.dynamic_viz import generate_dynamic_gif
    VIS_AVAILABLE = True

except ImportError:
    print("⚠️ Visualization modules not found. Running in text-only mode.")
    VIS_AVAILABLE = False
    
    class SystemRecorder:
        def __init__(self, path): pass
        def log_event(self, **kwargs): pass
        def save(self): pass

DEMO_DIR = "./demo"

def setup_env():
    if os.path.exists(DEMO_DIR): shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

def print_section(title):
    print(f"\n{'='*60}\n {title} \n{'='*60}")

def execute_turn(system, graph, agent, action):
    logs = system.execute_action(graph, agent, action)
    graph.id_alias.clear() 
    return logs

def demo_full_lifecycle():
    setup_env()
    trace_file = os.path.join(DEMO_DIR, "trace.json")
    recorder = SystemRecorder(trace_file) if VIS_AVAILABLE else None
    system = LegalSystem(persist_dir=DEMO_DIR, recorder=recorder)
    print_section("PHASE 1: 构建历史记忆")
    case_a_ctx = "被告原封不动地复制了原告的核心算法代码。"
    sg_a = ShadowGraph()
    
    execute_turn(system, sg_a, "plaintiff", """
        ADD_FACT("代码MD5一致")
        ADD_CLAIM("被告复制行为")
        ADD_CLAIM("侵犯著作权")
        SUPPORT(FACT_1, CLAIM_1)
        SUPPORT(CLAIM_1, CLAIM_2)
    """)
    
    for _,d in sg_a.graph.nodes(data=True):
        if "著作权" in d['content']: d['metadata']['is_root_claim'] = True
    
    system.learn(case_a_ctx, sg_a, "plaintiff", "CASE_WIN")
    print("✅ 案例 A 存入 (VALIDATED)")
    case_b_ctx = "被告离职后使用了原告的客户名单。"
    sg_b = ShadowGraph()
    
    execute_turn(system, sg_b, "plaintiff", """
        ADD_FACT("使用客户列表")
        ADD_CLAIM("侵犯商业秘密")
        SUPPORT(FACT_1, CLAIM_1)
    """)
    
    for _,d in sg_b.graph.nodes(data=True):
        if "商业秘密" in d['content']: d['metadata']['is_root_claim'] = True

    c_secret = [n for n,d in sg_b.graph.nodes(data=True) if "商业秘密" in d['content']][0]
    
    execute_turn(system, sg_b, "defendant", """
        ADD_FACT("公网可查")
        ADD_CLAIM("非秘密信息")
    """)

    c_not_secret = [n for n,d in sg_b.graph.nodes(data=True) if "非秘密" in d['content']][0]
    sg_b.add_edge(c_not_secret, c_secret, EdgeType.CONFLICT)
    system.learn(case_b_ctx, sg_b, "defendant", "CASE_LOSE")
    print("✅ 案例 B 存入 (DEFEATED)")
    print_section("PHASE 2: 新案审理与投影")
    current_context = "原告指控张三：1. 复制代码。2. 带走客户名单。"
    sg_live, _ = system.new_case(current_context)
    print(f"投影节点数: {len(sg_live.graph.nodes())}")
    print_section("PHASE 3: 对抗与完整性检查")
    print(">> Round 1 (Plaintiff)")
    
    actions_p1 = """
    ADD_CLAIM("要求停止代码侵权");
    ADD_CLAIM("要求赔偿商业秘密损失");
    """
    
    execute_turn(system, sg_live, "plaintiff", actions_p1)
    p_code_id = [n for n,d in sg_live.graph.nodes(data=True) if "停止代码" in d['content']][0]
    p_secret_id = [n for n,d in sg_live.graph.nodes(data=True) if "赔偿商业" in d['content']][0]
    sg_live.graph.nodes[p_code_id]['metadata']['is_root_claim'] = True
    sg_live.graph.nodes[p_secret_id]['metadata']['is_root_claim'] = True
    print(f"  Agent看到了: [{p_code_id}] 要求停止..., [{p_secret_id}] 要求赔偿...")
    print("\n>> Round 2 (Defendant)")
    facts = [n for n,d in sg_live.graph.nodes(data=True) if d['type'] == NodeType.FACT]
    
    if facts:
        f_id = facts[0]
        fail_log = execute_turn(system, sg_live, "defendant", f"SUPPORT({p_code_id}, {f_id})")
        print(f"  拦截结果 (循环): {fail_log[0]}")
    
    fail_log_2 = execute_turn(system, sg_live, "defendant", f"CHALLENGE(CLAIM_1, {p_secret_id})")
    print(f"  拦截结果 (无据): {fail_log_2[0]}")
    print("\n>> Round 3 (Defendant)")

    actions_d1 = f"""
    ADD_CLAIM("代码基于MIT开源协议");
    ADD_FACT("GitHub开源仓库记录");
    CHALLENGE(CLAIM_1, {p_code_id}, FACT_1); 
    """
    
    log_d = execute_turn(system, sg_live, "defendant", actions_d1)
    print(f"  执行指令: {actions_d1.strip()}")
    print(f"  执行结果: {log_d}")
    print_section("PHASE 4: 视图检查")
    print(sg_live.to_recursive_text())
    print_section("PHASE 5: 判决")
    d_mit_id = [n for n,d in sg_live.graph.nodes(data=True) if "MIT" in d['content']][0]
    ids = [d_mit_id, p_secret_id]
    print(f"  Judge认可: {ids}")
    final_sg = system.backprop.propagate(sg_live, ids)

    if recorder:
        recorder.log_event(
            step_name="Final State Check",
            shadow_graph=final_sg,
            message="Verdict Propagation Completed"
        )
    
    s_mit = final_sg.graph.nodes[d_mit_id]['status']
    s_code = final_sg.graph.nodes[p_code_id]['status']
    print(f"  MIT ({d_mit_id}): {s_mit}")
    print(f"  代码诉求 ({p_code_id}): {s_code}")
    
    if s_code == NodeStatus.DEFEATED: print("\n✅ 逻辑完美闭环：MIT抗辩成功击败了代码侵权诉求。")
    else: print("\n❌ 逻辑依然失败。")

    if recorder and VIS_AVAILABLE:
        recorder.save()
        gif_path = os.path.join(DEMO_DIR, "demo_showcase.gif")
        generate_dynamic_gif(trace_file, gif_path)
        print(f"\n✅ 动态演示 GIF 已生成: {gif_path}")

if __name__ == "__main__": demo_full_lifecycle()