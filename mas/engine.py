import json
import asyncio
from enum import Enum
from typing import Dict, Any
from metagpt.logs import logger
from .legal_system import LegalSystem
from .judge import LLMJudge
from .team import DebateTeam
from tools.initializer import CaseInitializer
from tools.graph_tool import GraphTool
from tools.fact_es_tool import FactEsTool
from tools.law_es_tool import LawEsTool
from .llm import GPTChat
from .config import SystemConfig
from .common import ShadowGraph
from mas.schema import AgentAction

class Turn(Enum):
    PLAINTIFF = "plaintiff"
    DEFENDANT = "defendant"

class DebateEngine:
    def __init__(self, config: SystemConfig, judge_config: Dict):
        self.cfg = config
        self.judge_config = judge_config
        self.legal_sys: LegalSystem = None
        self.p_team: DebateTeam = None
        self.d_team: DebateTeam = None
        self.graph: ShadowGraph = None
        self.raw_facts: str = ""
        self.current_turn: Turn = Turn.PLAINTIFF
        self.round_idx: int = 0
        self.max_rounds: int = 3
        self.is_finished: bool = False
        self.winner: str = "Unsettled"
        self.last_step_log: Dict[str, Any] = {}
        self.fact_es = None
        self.law_es = None

    async def setup(self, case_data_path: str, verbose: bool = False):
        logger.info(">>> [Engine] Setting up...")
        agent_llm = GPTChat(model_name=self.cfg.llm.model_name)
        
        judge_llm = GPTChat(
            model_name=self.cfg.judge.model_name,
            base_url=self.cfg.judge.base_url,
            api_key=self.cfg.judge.api_key
        )

        self.legal_sys = LegalSystem(config=self.cfg)
        self.legal_sys.judge = LLMJudge(llm=judge_llm)
        self.fact_es = FactEsTool(es_host=self.cfg.es.host, embedding_func=self.legal_sys.ef)
        self.law_es = LawEsTool(es_host=self.cfg.es.host, embedding_func=self.legal_sys.ef)
        graph_tool = GraphTool(legal_system=self.legal_sys, llm=agent_llm)
        with open(case_data_path, 'r', encoding='utf-8') as f: case_data = json.loads(f.readline())
        self.raw_facts = case_data.get("fact_finding", "")
        cause = case_data.get("cause", ["未知案由"])[0]
        initializer = CaseInitializer(agent_llm)
        init_res = await initializer.initialize(self.raw_facts, cause)
        self.graph, insights = self.legal_sys.new_case(self.raw_facts)
        graph_tool.set_current_graph(self.graph)
        fact_actions = []
        
        for fact_statement in init_res.fact_statements:
            fact_actions.append(AgentAction(
                action_type="add_fact",
                content=fact_statement,
                target_id=None,
                source_id=None,
                relation_type=None
            ))
        
        initial_actions = fact_actions + init_res.root_claim_actions
        
        if initial_actions:
            for action_obj in initial_actions:
                try:
                    logs = self.legal_sys.execute_action(self.graph, "System_Init", action_obj)
                    for log_msg in logs: logger.info(f"[System_Init] {log_msg}")
                
                except Exception as e: logger.error(f"[System_Init] Error executing initial action {action_obj.to_json()}: {e}")
        
        self.p_team = DebateTeam("plaintiff", init_res.plaintiff_persona, graph_tool, self.fact_es, self.law_es, agent_llm, insights, verbose=verbose)
        self.d_team = DebateTeam("defendant", init_res.defendant_persona, graph_tool, self.fact_es, self.law_es, agent_llm, insights, verbose=verbose)
        logger.info(">>> [Engine] Setup complete.")
        self.last_step_log = {"turn": "Setup", "action": "System Initialized", "details": f"{len(initial_actions)} initial claims/facts added."}

    async def step(self):
        if self.is_finished:
            logger.warning("[Engine] Debate is already finished. No more steps.")
            return
        
        is_plaintiff_turn = (self.current_turn == Turn.PLAINTIFF)
        
        if is_plaintiff_turn:
            if self.round_idx == 0: self.round_idx = 1
            logger.info(f"\n>>> [Engine] Round {self.round_idx}, Plaintiff's Turn...")
            team_to_run = self.p_team
            next_turn = Turn.DEFENDANT

        else:
            logger.info(f"\n>>> [Engine] Round {self.round_idx}, Defendant's Turn...")
            team_to_run = self.d_team
            next_turn = Turn.PLAINTIFF
            self.round_idx += 1

        turn_result = await team_to_run.run_turn(self.graph)

        self.last_step_log = {
            "turn": self.current_turn.value,
            "round": self.round_idx,
            "action": turn_result["summary"],
            "dialogue": turn_result["transcript"]
        }

        should_adjudicate = (not is_plaintiff_turn) or (self.round_idx >= self.max_rounds)

        if should_adjudicate:
            is_settled, winner = self.legal_sys.adjudicate(self.raw_facts, self.graph)
        
            if is_settled:
                self.is_finished = True
                self.winner = winner
                self.last_step_log["verdict"] = f"{winner} Wins!"
                logger.info(f">>> [Engine] Verdict Reached: {winner} Wins!")
            
            elif self.round_idx > self.max_rounds:
                self.is_finished = True
                self.winner = "Unsettled (Max rounds reached)"
                self.last_step_log["verdict"] = self.winner
                logger.info(f">>> [Engine] Max rounds reached.")

        if not self.is_finished:
            self.current_turn = next_turn
            if not is_plaintiff_turn: self.round_idx += 1

    def get_snapshot(self) -> Dict[str, str]:
        if not self.legal_sys: return {}

        return {
            "shadow_graph": self.graph,
            "insights_manager": self.legal_sys.insights,
            "task_layer": self.legal_sys.memory.task_layer,
            "last_log": self.last_step_log,
            "is_finished": self.is_finished,
            "winner": self.winner
        }
    
    async def close_resources(self):
        if self.fact_es:
            await self.fact_es.close()
            logger.info("[Engine] FactEsTool connection closed.")
        
        if self.law_es:
            await self.law_es.close()
            logger.info("[Engine] LawEsTool connection closed.")
