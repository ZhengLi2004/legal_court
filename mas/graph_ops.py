import re
from typing import List, Tuple, Optional, Any
from .common import ShadowGraph, NodeType, EdgeType
from .semantic_matcher import SemanticMatcher
# 解析 LLM 输出指令，执行图操作
class GraphExecutor:
    def __init__(self, graph: ShadowGraph, matcher: SemanticMatcher = None):
        self.graph = graph
        self.matcher = matcher
    # 执行单条指令并执行
    def apply_instruction(self, instruction: str, agent_id: str) -> str:
        instruction = instruction.strip()
        add_pattern = r'ADD_(FACT|LAW|CLAIM)\(\s*["\'](.*?)["\']\s*\)'
        add_match = re.match(add_pattern, instruction, re.DOTALL)
        
        if add_match:
            node_type_str, content = add_match.groups()
            
            try:
                node_type = NodeType[node_type_str]

                node_id = self.graph.add_node(
                    content=content, 
                    node_type=node_type, 
                    agent_id=agent_id,
                    matcher=self.matcher 
                )

                return f"Node Added/Merged: {node_id}"
            
            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"Error adding node: {e}"
        
        link_pattern = r'LINK\(\s*([a-zA-Z0-9_]+)\s*,\s*([a-zA-Z0-9_]+)\s*,\s*([a-zA-Z_]+)\s*\)'
        link_match = re.match(link_pattern, instruction)

        if link_match:
            src, tgt, type_str = link_match.groups()
            
            try:
                edge_type = EdgeType[type_str.upper()]
                self.graph.add_edge(src.strip(), tgt.strip(), edge_type)
                return f"Edge Added: {src}->{tgt}"
            
            except KeyError: return f"Error: Invalid Edge Type {type_str}"
            except ValueError as e: return f"Error: {str(e)}"

        challenge_pattern = r'CHALLENGE\(\s*([a-zA-Z0-9_]+)\s*,\s*([a-zA-Z0-9_]+)\s*\)'
        challenge_match = re.match(challenge_pattern, instruction)

        if challenge_match:
            src, tgt = challenge_match.groups()

            try:
                self.graph.add_edge(src.strip(), tgt.strip(), EdgeType.CONFLICT)
                return f"Challenge Added: {src}-|{tgt}"
            
            except Exception as e: return f"Error challenging: {e}"

        return "Error: Unknown Instruction"
    # 从一段 LLM 回复中提取多条指令、批量执行
    def execute_batch(self, llm_response: str, agent_id: str) -> List[str]:
        logs = []
        lines = llm_response.split('\n')

        for line in lines:
            clean_line = line.strip()
            clean_line = re.sub(r'^[\d\-\*\.]+\s*', '', clean_line)

            if any(cmd in clean_line for cmd in ["ADD_", "LINK(", "CHALLENGE("]):
                match = re.search(r'(ADD_|LINK\(|CHALLENGE\().*', clean_line, re.DOTALL)
                if match:
                    cmd = match.group(0)
                    cmd = re.sub(r'[.;]+$', '', cmd)                 
                    log = self.apply_instruction(cmd, agent_id)
                    logs.append(log)
        return logs