import copy
from typing import List, Dict, Tuple, Optional
from .common import ShadowGraph, NodeType, EdgeType, EdgeAddResult
from .semantic_matcher import SemanticMatcher
from .config import SystemConfig
from mas.schema import AgentAction, AgentActionType

class GraphExecutor:
    def __init__(self, graph: ShadowGraph, matcher: SemanticMatcher = None):
        self.graph = graph
        self.matcher = matcher

    def _apply_add_node(self, content: str, node_type: NodeType, agent_id: str, current_step: int = 0, metadata: Dict = None) -> Tuple[Optional[str], str]:
        try:
            if self.matcher:
                cfg = SystemConfig().dedup
                if node_type == NodeType.FACT: self.matcher.threshold = cfg.fact_threshold
                else: self.matcher.threshold = cfg.other_threshold

            node_id, is_new = self.graph.add_node(
                content=content,
                node_type=node_type,
                agent_id=agent_id,
                matcher=self.matcher,
                metadata=metadata or {}
            )

            self.graph.touch_nodes([node_id], current_step)
            type_cn = {NodeType.FACT: "事实", NodeType.LAW: "法条", NodeType.CLAIM: "主张"}.get(node_type, "节点")
            
            if is_new: return node_id, f"✅ [SUCCESS] 已添加新{type_cn}: {content} (ID: {node_id})"
            else: return node_id, f"⚠️ [NOTICE] 该{type_cn}已存在 (ID: {node_id})，已复用现有节点。"
                
        except Exception as e:  return None, f"Error: 无法添加节点 ({node_type.value} - {content}): {e}"

    def _apply_add_edge(self, source_id: str, target_id: str, edge_type: EdgeType, agent_id: str, current_step: int = 0) -> str:
        if not self.graph.graph.has_node(source_id): return f"Error: 源节点 '{source_id}' 未找到。"
        if not self.graph.graph.has_node(target_id): return f"Error: 目标节点 '{target_id}' 未找到。"

        try:
            if self.graph._get_node_type(target_id) == NodeType.LAW: return f"Error: 不能将关系连接到法条节点 '{target_id}'，法条是公理。"
            if edge_type == EdgeType.CONFLICT and self.graph._get_node_type(source_id) != NodeType.CLAIM: return f"Error: 只有 CLAIM 节点才能作为反驳关系的源节点。"
            result = self.graph.add_edge(source_id, target_id, edge_type)
            self.graph.touch_nodes([source_id, target_id], current_step)
            if result == EdgeAddResult.CREATED: return f"✅ [SUCCESS] {edge_type.value} 关系已添加: {source_id} -> {target_id}"
            elif result == EdgeAddResult.DUPLICATE: return f"ℹ️ [NOTICE] 关系已存在: {source_id} -> {target_id} ({edge_type.value})，无需重复添加。"
            
            elif result == EdgeAddResult.TYPE_CLASH:
                existing_data = self.graph.graph.get_edge_data(source_id, target_id)
                old_type = existing_data.get('type') if existing_data else "UNKNOWN"
                return f"❌ [REJECT] 关系冲突: {source_id} 与 {target_id} 之间已存在 {old_type} 关系，无法添加 {edge_type.value}。"
            
            elif result == EdgeAddResult.SELF_LOOP: return f"❌ [REJECT] 逻辑错误: 无法建立自环关系 (源节点与目标节点相同)。"
            else: return f"Error: 未知的边添加结果: {result}"
        
        except ValueError as ve: return f"Error: 添加 {edge_type.value} 关系失败: {ve}"
        except Exception as e: return f"Error: 添加 {edge_type.value} 关系时发生异常: {e}"

    def execute_batch(self, actions_batch: List[AgentAction], agent_id: str, current_step: int = 0) -> List[str]:
        logs = []
        original_nx_graph = copy.deepcopy(self.graph.graph)
        current_error = None

        try:
            for action in actions_batch:
                if action.action_type in [AgentActionType.CITE_FACT, AgentActionType.CITE_LAW]:
                    required_type = NodeType.FACT if action.action_type == AgentActionType.CITE_FACT else NodeType.LAW
                    type_str = "事实(FACT)" if required_type == NodeType.FACT else "法条(LAW)"
                    
                    if not action.source_id or not action.target_id:
                        current_error = ValueError(f"Error: {action.action_type} 必须同时提供 source_id 和 target_id。")
                        break
                    
                    if not self.graph.graph.has_node(action.source_id):
                        current_error = ValueError(f"Error: 指定的{type_str}节点 '{action.source_id}' 不存在。")
                        break
                    
                    actual_type = self.graph._get_node_type(action.source_id)
                    
                    if actual_type != required_type:
                        current_error = ValueError(f"Error: 节点 '{action.source_id}' 类型为 {actual_type}，但动作要求为 {required_type}。")
                        break
                    
                    if not self.graph.graph.has_node(action.target_id):
                        current_error = ValueError(f"Error: 目标观点节点 '{action.target_id}' 不存在。")
                        break
                    
                    res = self._apply_add_edge(action.source_id, action.target_id, EdgeType.SUPPORT, agent_id, current_step)
                    
                    if "Error" in res or "REJECT" in res:
                        current_error = ValueError(res)
                        break
                    
                    logs.append(res)

                elif action.action_type == AgentActionType.ADD_CLAIM:
                    node_id, log = self._apply_add_node(action.content, NodeType.CLAIM, agent_id, current_step, action.metadata)
                    
                    if node_id is None:
                        current_error = ValueError(log)
                        break
                    
                    logs.append(log)

                elif action.action_type in [AgentActionType.SUPPORT_CLAIM, AgentActionType.REBUT_CLAIM]:
                    is_support = (action.action_type == AgentActionType.SUPPORT_CLAIM)
                    edge_type = EdgeType.SUPPORT if is_support else EdgeType.CONFLICT
                    
                    if not action.target_id:
                        current_error = ValueError(f"Error: {action.action_type} 必须提供 target_id。")
                        break

                    source_id = action.source_id
                    
                    if source_id:
                        if not self.graph.graph.has_node(source_id):
                            current_error = ValueError(f"Error: 源节点 '{source_id}' 不存在。")
                            break
                        
                        if self.graph._get_node_type(source_id) != NodeType.CLAIM:
                            current_error = ValueError(f"Error: 逻辑推导/反驳的源头必须是 CLAIM 类型。")
                            break
                    
                    else:
                        if not action.content:
                            current_error = ValueError(f"Error: {action.action_type} 若不引用已有观点，必须提供 content。")
                            break
                        
                        source_id, log = self._apply_add_node(action.content, NodeType.CLAIM, agent_id, current_step, action.metadata)
                        logs.append(log)

                    res = self._apply_add_edge(source_id, action.target_id, edge_type, agent_id, current_step)
                    
                    if "Error" in res or "REJECT" in res:
                        current_error = ValueError(res)
                        break
                    
                    logs.append(res)
                
                else:
                    current_error = ValueError(f"Error: 收到未知的动作类型 {action.action_type}")
                    break

            if current_error: raise current_error
            return logs

        except Exception as e:
            self.graph.graph = original_nx_graph
            return [f"Error: 批量执行异常回滚。原因: {str(e)}"]