import shutil
import os
from mas.legal_system import LegalSystem
from vis.visualize_memory_state import snapshot_global_state
from vis.static_viz import GMemoryVisualizer

def run_evolution_demo():
    print("\n>>> Starting G-Memory Evolution Demo (5 Cases)...")
    root_dir = "./evolution_demo"
    storage_dir = os.path.join(root_dir, "storage")
    viz_dir = os.path.join(root_dir, "viz_output")
    if os.path.exists(root_dir): shutil.rmtree(root_dir)
    os.makedirs(viz_dir)
    system = LegalSystem(persist_dir=storage_dir)
    viz = GMemoryVisualizer(output_dir=os.path.join(viz_dir, "shadow_snapshots"))

    cases = [
        {
            "id": "case_01",
            "context": "被告人潜入邻居家拿走现金5000元，被抓获。",
            "p_action": 'ADD_FACT("被告人实施了入户盗窃"); ADD_LAW("刑法264条 盗窃罪"); LINK(FACT_1, LAW_2, SUPPORT)',
            "d_action": 'ADD_CLAIM("只是进去看看，没想偷"); CHALLENGE(CLAIM_3, FACT_1)',
            "winner": "plaintiff"
        },
        {
            "id": "case_02",
            "context": "被告人在商场偷拿了一部手机，价值3000元。", # 与 Case 1 相似
            "p_action": 'ADD_FACT("被告人秘密窃取他人财物"); ADD_LAW("刑法264条 盗窃罪"); LINK(FACT_1, LAW_2, SUPPORT)',
            "d_action": 'ADD_CLAIM("忘记付款了"); CHALLENGE(CLAIM_3, FACT_1)',
            "winner": "plaintiff"
        },
        {
            "id": "case_03", 
            "context": "原告借给被告10万元，被告逾期不还。", # 民事，与前两案不同
            "p_action": 'ADD_FACT("借贷关系成立"); ADD_LAW("民法典 借款合同"); LINK(FACT_1, LAW_2, SUPPORT)',
            "d_action": 'ADD_CLAIM("已口头延期"); CHALLENGE(CLAIM_3, FACT_1)',
            "winner": "plaintiff"
        },
        {
            "id": "case_04",
            "context": "被告人持刀拦路抢劫，抢走项链。", # 抢劫，与盗窃有部分语义重叠
            "p_action": 'ADD_FACT("被告人暴力抢劫"); ADD_LAW("刑法263条 抢劫罪"); LINK(FACT_1, LAW_2, SUPPORT)',
            "d_action": 'ADD_CLAIM("刀是切水果的"); CHALLENGE(CLAIM_3, FACT_1)',
            "winner": "plaintiff"
        },
        {
            "id": "case_05",
            "context": "被告人溜门入室，拿走了桌上的钱包。", # 与 Case 1 极度相似，测试投影
            "p_action": 'ADD_CLAIM("行为构成盗窃"); LINK(FACT_1, LAW_PROJECTED, SUPPORT)', # 假设利用了投影节点
            "d_action": 'ADD_CLAIM("认罪")',
            "winner": "plaintiff"
        }
    ]

    for i, case in enumerate(cases):
        print(f"\n=== Processing Round {i+1}: {case['id']} ===")
        print(f"Context: {case['context']}")
        sg_init, _ = system.new_case(case['context'])

        viz.draw_shadow_graph(
            sg_init, 
            filename=f"round_{i+1}_init_projection.png", 
            title=f"Round {i+1} Initial State\n(Projected from History)"
        )

        print(f"  > Snapshot taken: Initial ShadowGraph (Check for projected nodes)")
        system.execute_action(sg_init, "plaintiff", case['p_action'])
        system.execute_action(sg_init, "defendant", case['d_action'])
        system.learn(case['context'], sg_init, case['winner'], case['id'])
        print(f"  > Case learned and persisted.")
        snapshot_global_state(system, i+1, output_dir=os.path.join(viz_dir, "global_evolution"))

    print(f"\n✅ Evolution Demo Completed. Check outputs in {viz_dir}")

if __name__ == "__main__": run_evolution_demo()