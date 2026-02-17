"""Provides the logic for executing agent actions on the debate graph.

This module defines the `GraphExecutor` class, which is a dedicated component
for translating the structured `AgentAction` objects produced by agents into
actual modifications of the `ShadowGraph`. It handles action validation,
node and edge creation, and ensures that operations are transactional (i.e.,
a batch of actions either succeeds or fails as a whole).
"""

import copy
from typing import Dict, List, Optional, Tuple

import networkx as nx

from mas.core.schemas import AgentAction, AgentActionType
from tools.matcher import SemanticMatcher

from ..config import SystemConfig
from ..core.graph import EdgeAddResult, EdgeType, NodeType, ShadowGraph


class GraphExecutor:
    """Validates and executes a batch of AgentActions on a ShadowGraph.

    This class provides a safe and structured way to modify the debate graph.
    It first performs a static validation of all actions in a batch to catch
    errors early. If validation passes, it attempts to execute the batch
    transactionally.

    Attributes:
        graph: The `ShadowGraph` instance to be modified.
        matcher: An optional `SemanticMatcher` for node deduplication.
    """

    def __init__(self, graph: ShadowGraph, matcher: SemanticMatcher = None):
        """Initialize the GraphExecutor.

        Args:
            graph: The `ShadowGraph` to be operated on.
            matcher: The `SemanticMatcher` used for deduplicating new nodes.
        """
        self.graph = graph
        self.matcher = matcher

    def _validate_action_static(self, action: AgentAction, index: int) -> Optional[str]:
        """Perform static validation on a single AgentAction.

        This check verifies that the action has the required fields for its type,
        that specified node IDs exist in the graph, and that the node types are
        correct for the intended operation.

        Args:
            action: The `AgentAction` to validate.
            index: The index of the action in the batch, for error reporting.

        Returns:
            A string containing semicolon-separated validation error messages,
            or None if the action is valid.
        """
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

    def _check_would_create_cycle(self, source_id: str, target_id: str) -> bool:
        """Check if adding an edge would create a directed cycle (acyclic axiom).

        Checks if there is a path from target_id to source_id. If such a path exists,
        adding the edge source_id -> target_id would create a directed cycle,
        violating the acyclic axiom.

        Args:
            source_id: The source node ID.
            target_id: The target node ID.

        Returns:
            True if adding the edge would create a cycle, False otherwise.
        """
        if not self.graph.graph.has_node(source_id) or not self.graph.graph.has_node(
            target_id
        ):
            return False

        try:
            return nx.has_path(self.graph.graph, target_id, source_id)

        except nx.NetworkXError:
            return False

    def _apply_add_node(
        self,
        content: str,
        node_type: NodeType,
        agent_id: str,
        current_step: int = 0,
        metadata: Dict = None,
    ) -> Tuple[Optional[str], str]:
        """Apply the add_node operation to the graph.

        Args:
            content: The text content of the node.
            node_type: The `NodeType` of the node.
            agent_id: The ID of the agent performing the action.
            current_step: The current debate step, for metadata.
            metadata: Additional metadata for the node.

        Returns:
            A tuple containing the new node's ID and a log message.
        """
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
        """Apply the add_edge operation to the graph.

        Args:
            source_id: The ID of the source node.
            target_id: The ID of the target node.
            edge_type: The `EdgeType` of the edge.
            agent_id: The ID of the agent performing the action.
            current_step: The current debate step, for metadata.

        Returns:
            A log message describing the result of the operation.
        """
        try:
            if self._check_would_create_cycle(source_id, target_id):
                return f"⚠️ [警告] 添加边 {source_id} -> {target_id} 会产生循环，该操作已忽略（无环公理）。"

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
        """Execute a batch of actions atomically.

        First, it validates all actions. If any are invalid, it raises an
        error with a summary of all issues. If validation passes, it proceeds
        to execute the actions. If any action fails during execution, it reverts
        the graph to its state before the batch began and raises an error.

        Args:
            actions_batch: A list of `AgentAction` objects to execute.
            agent_id: The ID of the agent performing the actions.
            current_step: The current debate step index.

        Returns:
            A list of log strings detailing the outcome of each successful operation.

        Raises:
            ValueError: If static validation fails or if a runtime error occurs
                during execution.
        """
        logs = []
        validation_errors = self.validate_batch(actions_batch)

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

    def validate_batch(self, actions_batch: List[AgentAction]) -> List[str]:
        """Validate a batch of actions without mutating graph state.

        Args:
            actions_batch: Candidate actions to validate.

        Returns:
            A list of validation error messages. Empty means pass.
        """
        validation_errors: List[str] = []

        for i, action in enumerate(actions_batch):
            error = self._validate_action_static(action, i)

            if error:
                validation_errors.append(error)

        return validation_errors
