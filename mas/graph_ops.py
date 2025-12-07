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

    def apply_link(self, src_id: str, tgt_id: str, type_str: str) -> str:
        if not self.graph.graph.has_node(src_id): return f"Error: Source node '{src_id}' not found in graph."
        if not self.graph.graph.has_node(tgt_id): return f"Error: Target node '{tgt_id}' not found in graph."
            
        try:
            edge_type = EdgeType[type_str.upper()]
            self.graph.add_edge(src_id, tgt_id, edge_type=edge_type)
            return f"Edge Added: {src_id} -> {tgt_id}"
        
        except Exception as e: return f"Error linking nodes: {e}"
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
                count = len([k for k in self.graph.id_alias if k.startswith(node_type_str)]) + 1
                local_key = f"{node_type_str}_{count}"
                real_id = self.apply_add(clean_cmd, agent_id)
                
                if not real_id.startswith("Error"):
                    self.graph.id_alias[local_key] = real_id
                    logs.append(f"Aliased {local_key} -> {real_id}")
                
                else: logs.append(real_id)
                continue

            link_pattern = r'LINK\(\s*([a-zA-Z0-9_]+)\s*,\s*([a-zA-Z0-9_]+)\s*,\s*([a-zA-Z_]+)\s*\)'
            link_match = re.match(link_pattern, clean_cmd)
            
            if link_match:
                src_alias, tgt_alias, type_str = link_match.groups()
                real_src = self.graph.id_alias.get(src_alias, src_alias)
                real_tgt = self.graph.id_alias.get(tgt_alias, tgt_alias)
                log = self.apply_link(real_src, real_tgt, type_str)
                logs.append(log)
                continue

            challenge_pattern = r'CHALLENGE\(\s*([a-zA-Z0-9_]+)\s*,\s*([a-zA-Z0-9_]+)\s*\)'
            challenge_match = re.match(challenge_pattern, clean_cmd)
            
            if challenge_match:
                src_alias, tgt_alias = challenge_match.groups()
                real_src = self.graph.id_alias.get(src_alias, src_alias)
                real_tgt = self.graph.id_alias.get(tgt_alias, tgt_alias)
                log = self.apply_link(real_src, real_tgt, "CONFLICT")
                logs.append(log)
                continue
            
            logs.append(f"Error: Unknown instruction '{clean_cmd}'")
            
        return logs