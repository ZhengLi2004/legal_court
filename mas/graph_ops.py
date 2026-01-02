import copy
from typing import Dict, List, Optional, Tuple

from mas.schema import AgentAction, AgentActionType

from .common import EdgeAddResult, EdgeType, NodeType, ShadowGraph
from .config import SystemConfig
from .semantic_matcher import SemanticMatcher


class GraphExecutor:
    def __init__(self, graph: ShadowGraph, matcher: SemanticMatcher = None):
        self.graph = graph
        self.matcher = matcher

    def _validate_action_static(self, action: AgentAction, index: int) -> Optional[str]:
        errors = []
        prefix = f"第 {index + 1} 个动作 ({action.action_type.value})"

        if action.action_type in [AgentActionType.CITE_FACT, AgentActionType.CITE_LAW]:
            required_src_type = (
                NodeType.FACT
                if action.action_type == AgentActionType.CITE_FACT
                else NodeType.LAW
            )

            if not action.source_id:
                errors.append(
                    f"{prefix}: 缺少 'source_id'。引用动作必须指定一个已存在的依据节点(FACT/LAW)。"
                )

            elif not self.graph.graph.has_node(action.source_id):
                errors.append(f"{prefix}: 源节点 '{action.source_id}' 不存在。")

            else:
                src_type = self.graph._get_node_type(action.source_id)

                if src_type != required_src_type:
                    errors.append(
                        f"{prefix}: 源节点类型必须是 {required_src_type.value}，但实际是 {src_type.value}。"
                    )

            if action.target_id:
                if not self.graph.graph.has_node(action.target_id):
                    errors.append(f"{prefix}: 目标节点 '{action.target_id}' 不存在。")

                else:
                    tgt_type = self.graph._get_node_type(action.target_id)

                    if tgt_type != NodeType.CLAIM:
                        errors.append(
                            f"{prefix}: 引用的目标必须是 CLAIM 节点，但实际是 {tgt_type.value}。"
                        )

            else:
                if not action.content:
                    errors.append(
                        f"{prefix}: 当基于引用创建新观点时（未指定 target_id），'content' 字段不能为空。"
                    )

        elif action.action_type in [
            AgentActionType.SUPPORT_CLAIM,
            AgentActionType.REBUT_CLAIM,
        ]:
            if not action.target_id:
                errors.append(
                    f"{prefix}: 缺少 'target_id'。逻辑动作必须指向一个已存在的 CLAIM 节点。"
                )

            elif not self.graph.graph.has_node(action.target_id):
                errors.append(f"{prefix}: 目标节点 '{action.target_id}' 不存在。")

            else:
                tgt_type = self.graph._get_node_type(action.target_id)

                if tgt_type != NodeType.CLAIM:
                    errors.append(
                        f"{prefix}: 逻辑动作的目标必须是 CLAIM 节点，但实际是 {tgt_type.value}。"
                    )

            if action.source_id:
                if not self.graph.graph.has_node(action.source_id):
                    errors.append(f"{prefix}: 源节点 '{action.source_id}' 不存在。")

                else:
                    src_type = self.graph._get_node_type(action.source_id)

                    if src_type != NodeType.CLAIM:
                        errors.append(
                            f"{prefix}: 逻辑动作的源节点必须是 CLAIM 节点，但实际是 {src_type.value}。"
                        )

                    if action.target_id and action.source_id == action.target_id:
                        errors.append(
                            f"{prefix}: 源节点和目标节点不能相同（禁止自环）。"
                        )

            else:
                if not action.content:
                    errors.append(
                        f"{prefix}: 当提出新观点进行支持/反驳时（未指定 source_id），'content' 字段不能为空。"
                    )

        else:
            errors.append(f"{prefix}: 未知的动作类型。")

        return "; ".join(errors) if errors else None

    def _apply_add_node(
        self,
        content: str,
        node_type: NodeType,
        agent_id: str,
        current_step: int = 0,
        metadata: Dict = None,
    ) -> Tuple[Optional[str], str]:
        try:
            if self.matcher:
                cfg = SystemConfig().dedup

                if node_type == NodeType.FACT:
                    self.matcher.threshold = cfg.fact_threshold

                else:
                    self.matcher.threshold = cfg.other_threshold

            node_id, is_new = self.graph.add_node(
                content=content,
                node_type=node_type,
                agent_id=agent_id,
                matcher=self.matcher,
                metadata=metadata or {},
            )

            self.graph.touch_nodes([node_id], current_step)

            type_cn = {
                NodeType.FACT: "事实",
                NodeType.LAW: "法条",
                NodeType.CLAIM: "主张",
            }.get(node_type, "节点")

            if is_new:
                return (
                    node_id,
                    f"✅ [成功] 已添加新{type_cn}: {content[:20]}... (ID: {node_id})",
                )

            else:
                return (
                    node_id,
                    f"ℹ️ [提示] 该{type_cn}已存在 (ID: {node_id})，已复用现有节点。",
                )

        except Exception as e:
            return None, f"错误: 无法添加节点 ({node_type.value}): {e}"

    def _apply_add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        agent_id: str,
        current_step: int = 0,
    ) -> str:
        try:
            result = self.graph.add_edge(source_id, target_id, edge_type)
            self.graph.touch_nodes([source_id, target_id], current_step)

            if result == EdgeAddResult.CREATED:
                return f"✅ [成功] {edge_type.value} 关系已添加: {source_id} -> {target_id}"

            elif result == EdgeAddResult.DUPLICATE:
                return f"ℹ️ [提示] 关系已存在: {source_id} -> {target_id}"

            elif result == EdgeAddResult.TYPE_CLASH:
                return f"❌ [拒绝] 关系冲突: {source_id} -> {target_id}"

            elif result == EdgeAddResult.SELF_LOOP:
                return "⚠️ [警告] 自环操作被忽略。"

            else:
                return f"错误: 未知的边添加结果: {result}"

        except Exception as e:
            return f"错误: 添加关系异常: {e}"

    def execute_batch(
        self, actions_batch: List[AgentAction], agent_id: str, current_step: int = 0
    ) -> List[str]:
        logs = []
        validation_errors = []

        for i, action in enumerate(actions_batch):
            error = self._validate_action_static(action, i)

            if error:
                validation_errors.append(error)

        if validation_errors:
            error_summary = "\n".join(validation_errors)
            raise ValueError(f"批量操作校验失败，请修正以下错误:\n{error_summary}")

        original_nx_graph = copy.deepcopy(self.graph.graph)

        try:
            for action in actions_batch:
                final_source_id = None
                final_target_id = None
                edge_type = EdgeType.SUPPORT

                if action.action_type in [
                    AgentActionType.CITE_FACT,
                    AgentActionType.CITE_LAW,
                ]:
                    final_source_id = action.source_id

                    if action.target_id:
                        final_target_id = action.target_id

                    else:
                        final_target_id, log = self._apply_add_node(
                            action.content,
                            NodeType.CLAIM,
                            agent_id,
                            current_step,
                            action.metadata,
                        )

                        logs.append(log)

                    edge_type = EdgeType.SUPPORT

                elif action.action_type in [
                    AgentActionType.SUPPORT_CLAIM,
                    AgentActionType.REBUT_CLAIM,
                ]:
                    final_target_id = action.target_id

                    if action.source_id:
                        final_source_id = action.source_id

                    else:
                        final_source_id, log = self._apply_add_node(
                            action.content,
                            NodeType.CLAIM,
                            agent_id,
                            current_step,
                            action.metadata,
                        )

                        logs.append(log)

                    edge_type = (
                        EdgeType.CONFLICT
                        if action.action_type == AgentActionType.REBUT_CLAIM
                        else EdgeType.SUPPORT
                    )

                if final_source_id and final_target_id:
                    if final_source_id == final_target_id:
                        logs.append(
                            f"⚠️ [注意] 节点去重导致了自环 (ID: {final_source_id})，该连边操作已跳过。"
                        )

                    else:
                        res = self._apply_add_edge(
                            final_source_id,
                            final_target_id,
                            edge_type,
                            agent_id,
                            current_step,
                        )

                        logs.append(res)

                else:
                    raise ValueError(f"无法解析动作所需的 ID: {action.action_type}")

            return logs

        except Exception as e:
            self.graph.graph = original_nx_graph
            raise ValueError(f"运行时执行错误 (已回滚): {str(e)}")
