import re
from typing import List
from .common import ShadowGraph, NodeType, EdgeType
from .semantic_matcher import SemanticMatcher
from .config import SystemConfig
# 解析 LLM 输出指令，执行图操作
class GraphExecutor:
    def __init__(self, graph: ShadowGraph, matcher: SemanticMatcher = None):
        self.graph = graph
        self.matcher = matcher
    # 执行单条指令并执行
    def apply_add(self, instruction: str, agent_id: str) -> str:
        add_pattern = r'ADD_(FACT|LAW|CLAIM)\(["\'](.*?)["\']\)'
        add_match = re.match(add_pattern, instruction, re.DOTALL)
        
        if add_match:
            node_type_str, content = add_match.groups()
            
            try:
                node_type = NodeType[node_type_str]
                
                if self.matcher:
                    cfg = SystemConfig().dedup
                    if node_type == NodeType.FACT: self.matcher.threshold = cfg.fact_threshold
                    else: self.matcher.threshold = cfg.other_threshold

                node_id = self.graph.add_node(
                    content=content, node_type=node_type, agent_id=agent_id, matcher=self.matcher
                )
                
                return node_id
            
            except Exception as e: return f"Error: {e}"
        
        return "Error: Unknown ADD format"

    def apply_support(self, src_alias: str, tgt_alias:str) -> str:
        real_src = self.graph.id_alias.get(src_alias, src_alias)
        real_tgt = self.graph.id_alias.get(tgt_alias, tgt_alias)
        if not self.graph.graph.has_node(real_src): return f"Error: Source '{src_alias}' not found."
        if not self.graph.graph.has_node(real_tgt): return f"Error: Target '{tgt_alias}' not found."

        try:
            self.graph.add_edge(real_src, real_tgt, EdgeType.SUPPORT)
            return f"Support Added: {real_src} -> {real_tgt}"
        
        except ValueError as ve: return f"Error adding support: {ve}"
        except Exception as e: return f"Error adding support: {e}"

    def apply_challenge(self, attacker_alias: str, target_alias: str, evidence_alias: str = None) -> str:
        real_attacker = self.graph.id_alias.get(attacker_alias, attacker_alias)
        real_target = self.graph.id_alias.get(target_alias, target_alias)
        real_evidence = self.graph.id_alias.get(evidence_alias, evidence_alias) if evidence_alias else None
        if not self.graph.graph.has_node(real_attacker): return f"Error: Attacker '{attacker_alias}' not found."
        if not self.graph.graph.has_node(real_target): return f"Error: Target '{target_alias}' not found."
        if not real_evidence: return "Error: CHALLENGE rejected. No evidence cited."
        if not self.graph.is_valid_evidence(real_evidence): return f"Error: Cited evidence '{evidence_alias}' is not valid (must be FACT or LAW)."

        try:
            self.graph.add_edge(real_evidence, real_attacker, EdgeType.SUPPORT)
            self.graph.add_edge(real_attacker, real_target, EdgeType.CONFLICT)

        except ValueError as ve: return f"Error in Challenge topology: {ve}"
        return f"Challenge Added: {real_attacker} attacks {real_target} (based on {real_evidence})"
    # 从一段 LLM 回复中提取多条指令、批量执行
    def execute_batch(self, llm_response: str, agent_id: str) -> List[str]:
        logs = []
        instructions = []
        raw_lines = llm_response.split('\n')
        for line in raw_lines: instructions.extend([part.strip() for part in line.split(';') if part.strip()])

        for cmd_text in instructions:
            clean_cmd = re.sub(r'^[\d\-\*\.]+\s*', '', cmd_text).strip()
            if not clean_cmd: continue
            add_pattern = r'ADD_(FACT|LAW|CLAIM)\(["\'](.*?)["\']\)'
            add_match = re.match(add_pattern, clean_cmd, re.DOTALL)
            
            if add_match:
                node_type_str = add_match.group(1)
                count = len([k for k in self.graph.id_alias if k.startswith(f"{node_type_str}")]) + 1
                local_key = f"{node_type_str}_{count}"
                real_id = self.apply_add(clean_cmd, agent_id)
                
                if not real_id.startswith("Error"):
                    self.graph.id_alias[local_key] = real_id
                    logs.append(f"Aliased {local_key} -> {real_id}")
                
                else: logs.append(real_id)
                continue

            support_pattern = r'SUPPORT\(\s*([^,)]+?)\s*,\s*([^,)]+?)\s*\)'
            support_match = re.match(support_pattern, clean_cmd)
            
            if support_match:
                src, tgt = support_match.groups()
                logs.append(self.apply_support(src, tgt))
                continue

            challenge_pattern = r'CHALLENGE\(\s*([^,)]+?)\s*,\s*([^,)]+?)(?:\s*,\s*([^,)]+?))?\s*\)'
            challenge_match = re.match(challenge_pattern, clean_cmd)
            
            if challenge_match:
                src_alias, tgt_alias, evidence_alias = challenge_match.groups()
                log = self.apply_challenge(src_alias, tgt_alias, evidence_alias)
                logs.append(log)
                continue
            
            logs.append(f"Error: Unknown instruction '{clean_cmd}'")
            
        return logs