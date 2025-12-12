import asyncio
import json
import os
from metagpt.logs import logger
from mas.legal_system import LegalSystem
from tools.initializer import CaseInitializer
from mas.team import DebateTeam
from tools.graph_tool import GraphTool
from tools.fact_es_tool import FactEsTool
from tools.law_es_tool import LawEsTool
from mas.llm import GPTChat
from mas.config import SystemConfig
DATA_FILE = "data/sampling/cleaned_samples.jsonl"

async def main():
    logger.info(">>> Initializing System Components...")
    cfg = SystemConfig()
    llm = GPTChat(model_name=cfg.llm.model_name)
    legal_sys = LegalSystem(persist_dir="./demo", config=cfg)
    fact_es = FactEsTool(es_host=cfg.es.host, embedding_func=legal_sys.ef)
    law_es = LawEsTool(es_host=cfg.es.host, embedding_func=legal_sys.ef)
    graph_tool = GraphTool(legal_system=legal_sys, llm=llm)
    
    if not os.path.exists(DATA_FILE):
        logger.error(f"Data file not found: {DATA_FILE}")
        return

    logger.info(f">>> Loading Case Data from {DATA_FILE} (JSONL format)...")

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        
        if not first_line:
            logger.error("Data file is empty!")
            return
        
        try: case_data = json.loads(first_line)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSONL line: {e}")
            return

    raw_facts = case_data.get("fact_finding", "")
    cause = case_data.get("cause", ["未知案由"])[0]
    logger.info(">>> Running Case Initializer...")
    initializer = CaseInitializer(llm)
    init_res = await initializer.initialize(raw_facts, cause)
    logger.info(">>> Seeding Graph (G0)...")
    sg, _ = legal_sys.new_case(raw_facts)
    graph_tool.set_current_graph(sg)
    seed_script = "\n".join(init_res.fact_actions + init_res.root_claim_actions)

    if seed_script:
        logger.info(f"Executing Seed Script:\n{seed_script}")
        logs = legal_sys.execute_action(sg, "System_Init", seed_script)
        logger.info(f"Seed Logs: {logs}")

    logger.info(">>> Assembling Debate Teams...")

    p_team = DebateTeam(
        side="plaintiff",
        persona=init_res.plaintiff_persona,
        graph_tool=graph_tool,
        fact_es=fact_es,
        law_es=law_es,
        llm=llm
    )

    d_team = DebateTeam(
        side="defendant",
        persona=init_res.defendant_persona,
        graph_tool=graph_tool,
        fact_es=fact_es,
        law_es=law_es,
        llm=llm
    )

    MAX_ROUNDS = 3
    round_idx = 0
    logger.info("\n" + "="*20 + " DEBATE START " + "="*20)

    while round_idx < MAX_ROUNDS:
        round_idx += 1
        logger.info(f"\n>>> Round {round_idx}/{MAX_ROUNDS}")
        p_log = await p_team.run_turn(sg)
        logger.info(f"[Plaintiff Action]: {p_log}")
        is_settled, winner = legal_sys.adjudicate(raw_facts, sg)
        
        if is_settled:
            logger.info(f">>> Verdict Reached: {winner} Wins!")
            break

        d_log = await d_team.run_turn(sg)
        logger.info(f"[Defendant Action]: {d_log}")
        is_settled, winner = legal_sys.adjudicate(raw_facts, sg)
        
        if is_settled:
            logger.info(f">>> Verdict Reached: {winner} Wins!")
            break

    logger.info("\n" + "="*20 + " DEBATE END " + "="*20)
    final_context = sg.to_recursive_text()
    print("\n[Final Debate Graph]:\n")
    print(final_context)
    await fact_es.close()
    await law_es.close()

if __name__ == "__main__": asyncio.run(main())