import copy
from typing import List, Dict
from .common import ShadowGraph, NodeType, EdgeType
from .semantic_matcher import SemanticMatcher
from .config import SystemConfig
from mas.schema import AgentAction, AgentActionType
# 解析 LLM 输出指令，执行图操作
class GraphExecutor:
    def __init__(self, graph: ShadowGraph, matcher: SemanticMatcher = None):
        self.graph = graph
        self.matcher = matcher

    def _apply_add_node(self, content: str, node_type: NodeType, agent_id: str, current_step: int = 0) -> str:
        try:
            if self.matcher:
                cfg = SystemConfig().dedup
                if node_type == NodeType.FACT: self.matcher.threshold = cfg.fact_threshold
                else: self.matcher.threshold = cfg.other_threshold

            meta = {}
            if current_step == 1 and node_type == NodeType.CLAIM: meta["is_root_claim"] = True

            node_id = self.graph.add_node(
                content=content,
                node_type=node_type,
                agent_id=agent_id,
                matcher=self.matcher,
                metadata=meta
            )
            self.graph.touch_nodes([node_id], current_step)
            return node_id
        except Exception as e: return f"Error: 无法添加节点 ({node_type.value} - {content}): {e}"

    def _apply_add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, agent_id: str, current_step: int = 0) -> str:
        if not self.graph.graph.has_node(source_id): return f"Error: 源节点 '{source_id}' 未找到。"
        if not self.graph.graph.has_node(target_id): return f"Error: 目标节点 '{target_id}' 未找到。"

        try:
            if self.graph._get_node_type(target_id) == NodeType.LAW: return f"Error: 不能将关系连接到法条节点 '{target_id}'，法条是公理。"
            if edge_type == EdgeType.CONFLICT and self.graph._get_node_type(source_id) != NodeType.CLAIM: return f"Error: 只有 CLAIM 节点才能作为反驳关系的源节点。"
            self.graph.add_edge(source_id, target_id, edge_type)
            self.graph.touch_nodes([source_id, target_id], current_step)
            return f"{edge_type.value} 关系已添加: {source_id} -> {target_id}"
        
        except ValueError as ve: return f"Error: 添加 {edge_type.value} 关系失败: {ve}"
        except Exception as e: return f"Error: 添加 {edge_type.value} 关系时发生异常: {e}"

    def execute_batch(self, actions_batch: List[AgentAction], agent_id: str, current_step: int = 0) -> List[str]:
        logs = []
        original_nx_graph = copy.deepcopy(self.graph.graph)
        current_error = None

        try:
            for action in actions_batch:
                if action.action_type == AgentActionType.ADD_FACT:
                    if agent_id not in [None, "System_Init"]:
                        current_error = ValueError(f"Error: 代理 '{agent_id}' 尝试在辩论阶段添加新事实 '{action.content}'。事实只能在初始化阶段添加。")
                        break
                    
                    node_id = self._apply_add_node(action.content, NodeType.FACT, agent_id, current_step)
                    
                    if node_id.startswith("Error"):
                        current_error = ValueError(node_id)
                        break
                    
                    logs.append(f"添加事实: {action.content} (ID: {node_id})")

                elif action.action_type == AgentActionType.ADD_CLAIM:
                    node_id = self._apply_add_node(action.content, NodeType.CLAIM, agent_id, current_step)
                    
                    if node_id.startswith("Error"):
                        current_error = ValueError(node_id)
                        break
                    
                    logs.append(f"添加主张: {action.content} (ID: {node_id})")

                elif action.action_type == AgentActionType.CITE_LAW:
                    node_id = self._apply_add_node(action.content, NodeType.LAW, agent_id, current_step)
                    
                    if node_id.startswith("Error"):
                        current_error = ValueError(node_id)
                        break
                    
                    logs.append(f"引用法条: {action.content} (ID: {node_id})")
                    
                    if action.target_id and action.relation_type == EdgeType.SUPPORT:
                        res = self._apply_add_edge(node_id, action.target_id, EdgeType.SUPPORT, agent_id, current_step)
                        
                        if res.startswith("Error"):
                            current_error = ValueError(res)
                            break
                        
                        logs.append(res)
                    
                    elif action.target_id and action.relation_type == EdgeType.CONFLICT:
                        current_error = ValueError("Error: 法条不能直接反驳主张，它必须通过支持一个反驳方来进行。")
                        break
            
            if current_error: raise current_error

            for action in actions_batch:
                if action.action_type == AgentActionType.REBUT_CLAIM:
                    if not action.target_id:
                        current_error = ValueError("REBUT_CLAIM 操作需要提供 target_id (被反驳的主张ID)。")
                        break
                    
                    source_node_for_conflict = action.source_id
                    
                    if not action.source_id:
                        implicit_claim_id = self._apply_add_node(action.content, NodeType.CLAIM, agent_id, current_step)
                        
                        if implicit_claim_id.startswith("Error"):
                            current_error = ValueError(implicit_claim_id)
                            break
                        
                        logs.append(f"为反驳添加隐式主张: {action.content} (ID: {implicit_claim_id})")
                        source_node_for_conflict = implicit_claim_id
                    
                    res = self._apply_add_edge(source_node_for_conflict, action.target_id, EdgeType.CONFLICT, agent_id, current_step)
                    
                    if res.startswith("Error"):
                        current_error = ValueError(res)
                        break
                    
                    logs.append(res)

            if current_error: raise current_error
            return logs

        except Exception as e:
            self.graph.graph = original_nx_graph
            error_msg = f"Error: 执行批量操作失败。已回滚。原因: {str(e)}"
            return [error_msg]