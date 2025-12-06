import shutil
import os
from mas.legal_system import LegalSystem
from mas.common import ShadowGraph
from vis.recorder import SystemRecorder
from vis.dynamic_viz import generate_dynamic_gif

def run_demo():
    print("\n>>> Starting G-Memory Demo...")
    root_dir = "./demo"
    storage_dir = os.path.join(root_dir, "storage")
    trace_file = os.path.join(root_dir, "trace.json")
    gif_file = os.path.join(root_dir, "demo.gif")
    if os.path.exists(root_dir): shutil.rmtree(root_dir)
    os.makedirs(storage_dir)
    recorder = SystemRecorder(trace_file)
    system = LegalSystem(persist_dir=storage_dir, recorder=recorder)
    seed_context = "被告人李四注册空壳公司，伪造购销合同，向银行申请贷款500万元后转移资金。"
    seed_sg = ShadowGraph()
    system.execute_action(seed_sg, "plaintiff", """
        ADD_FACT("注册空壳公司并伪造合同")
        ADD_FACT("转移贷款资金")
        ADD_LAW("刑法193条 贷款诈骗罪")
        ADD_LAW("刑法224条 合同诈骗罪")
        ADD_CLAIM("行为人具有非法占有目的")
        ADD_CLAIM("属于数罪竞合，择一重处")
        LINK(FACT_1, LAW_1, SUPPORT)
        LINK(FACT_1, LAW_2, SUPPORT)
        LINK(FACT_1, CLAIM_1, SUPPORT)
        LINK(FACT_2, CLAIM_1, SUPPORT)
        LINK(CLAIM_1, CLAIM_2, SUPPORT)
    """)
    system.learn(seed_context, seed_sg, "plaintiff", "case_seed_fraud")
    print("--- Memory Seeded with Complex Fraud Case ---")
    context = "被告人王五编造元宇宙养猪项目，承诺高额回报，向公众非法集资后挥霍。"
    sg_live, _ = system.new_case(context)
    print("\n--- Plaintiff's Turn 1 ---")
    p_action_1 = 'ADD_FACT("编造虚假项目骗取款项")'
    system.execute_action(sg_live, "plaintiff", p_action_1)
    print("\n--- Defendant's Turn 1 ---")
    d_action_1 = 'ADD_CLAIM("项目真实，只是市场原因失败")'
    system.execute_action(sg_live, "defendant", d_action_1)
    print("\n--- Plaintiff's Turn 2 ---")
    p_action_2 = 'ADD_FACT("被告人将资金用于个人奢侈品消费")'
    system.execute_action(sg_live, "plaintiff", p_action_2)
    system.learn(context, sg_live, "plaintiff", "case_test_hybrid")
    recorder.save()
    generate_dynamic_gif(trace_file, gif_file, duration=2.5)
    print(f"\n✅ Ultimate Demo Completed. GIF saved to {gif_file}")

if __name__ == "__main__": run_demo()