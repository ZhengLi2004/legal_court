import re
from typing import List, Tuple, Optional
from .common import ShadowGraph, NodeType, EdgeType
# 解析 LLM 输出指令，执行图操作
class GraphExecutor:
    def __init__(self, graph: ShadowGraph): self.graph = graph
    # 执行单条指令并执行
    def apply_instruction(self, instruction: str) -> str:
        instruction = instruction.strip()
        add_pattern = r'ADD_(FACT|LAW|CLAIM)\(["\'](.*?)["\']\)'
        add_match = re.match(add_pattern, instruction)
        
        if add_match:
            node_type_str, content = add_match.groups()
            node_type = NodeType[node_type_str]
            # TODO: 这里的 agent_id 暂时 hardcode，后续应从上下文传入
            node_id = self.graph.add_node(content, node_type, agent_id="commander")
            return f"Node Added: {node_id}"
        
        link_pattern = r'LINK\((.*?),\s*(.*?),\s*(.*?)\)'
        link_match = re.match(link_pattern, instruction)

        if link_match:
            src, tgt, type_str = link_match.groups()
            
            try:
                edge_type = EdgeType[type_str.upper()]
                self.graph.add_edge(src.strip(), tgt.strip(), edge_type)
                return f"Edge Added: {src}->{tgt}"
            
            except KeyError: return f"Error: Invalid Edge Type {type_str}"
            except ValueError as e: return f"Error: {str(e)}"

        return "Error: Unknown Instruction"
    # 从一段 LLM 回复中提取多条指令、批量执行
    def execute_batch(self, llm_response: str) -> List[str]:
        logs = []
        lines = llm_response.split('\n')

        for line in lines:
            if "ADD_" in line or "LINK(" in line:
                clean_line = line.strip()
                match = re.search(r'(ADD_|LINK\().*', clean_line)

                if match:
                    cmd = match.group(0)
                    if cmd.endswith('.'): cmd = cmd[:-1]
                    log = self.apply_instruction(cmd)
                    logs.append(log)

        return logs