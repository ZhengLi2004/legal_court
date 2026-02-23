"""Provides a high-level interface for agents to interact with the debate graph.

This module defines the `GraphTool`, which serves as the primary tool for
agents to modify the `ShadowGraph`. It abstracts the lower-level graph
operations and ensures that agent intentions are translated into valid graph
modifications.
"""

from typing import List, Tuple, Union

from mas.analysis.executor import GraphExecutor
from mas.core.graph import NodeType, ShadowGraph
from mas.core.schemas import AgentAction
from mas.core.system import LegalSystem
from tools.llm import LLMCallable


class GraphTool:
    """A tool for agents to validate/apply actions on the debate graph.

    This class holds the current state of the debate graph and provides methods
    for agents to apply changes. It uses a `GraphExecutor` to handle the
    underlying logic of validating and applying actions. It also has a special
    method for workers to inject new law nodes directly into the graph.

    Attributes:
        system: The `LegalSystem` object, providing access to system-wide resources.
        llm: The language model client.
        current_graph: The `ShadowGraph` instance representing the current debate state.
    """

    def __init__(self, legal_system: LegalSystem, llm: LLMCallable):
        """Initialize the GraphTool.

        Args:
            legal_system: The main `LegalSystem` instance.
            llm: The language model client.
        """
        self.system = legal_system
        self.llm = llm
        self.current_graph: ShadowGraph = None

    def set_current_graph(self, graph: ShadowGraph):
        """Update the tool with the latest version of the debate graph.

        This is typically called at the beginning of each agent's turn.

        Args:
            graph: The `ShadowGraph` for the current turn.
        """
        self.current_graph = graph

    def _normalize_actions(
        self, actions: Union[AgentAction, List[AgentAction]]
    ) -> List[AgentAction]:
        """Normalize one or many actions into a validated list."""
        if isinstance(actions, AgentAction):
            return [actions]

        if isinstance(actions, list) and all(
            isinstance(a, AgentAction) for a in actions
        ):
            return actions

        raise ValueError("无效的动作格式。")

    def validate_actions(
        self, actions: Union[AgentAction, List[AgentAction]]
    ) -> Tuple[bool, List[str]]:
        """Validate actions against graph constraints without applying them."""
        try:
            if not self.current_graph:
                return False, ["当前图谱上下文未设置。"]

            actions_list = self._normalize_actions(actions)

            executor = GraphExecutor(
                self.current_graph,
                matcher=self.system.dedup_matcher,
                dedup_thresholds=self.system.dedup_thresholds,
            )

            errors = executor.validate_batch(actions_list)

            if errors:
                return False, errors

            return True, []

        except Exception as e:
            return False, [f"GraphTool 校验异常: {str(e)}"]

    async def apply_actions(
        self, agent_id: str, actions: Union[AgentAction, List[AgentAction]]
    ) -> str:
        """Apply validated actions to graph state."""
        try:
            if not self.current_graph:
                return "REJECT: 当前图谱上下文未设置。"

            actions_list = self._normalize_actions(actions)

            logs = self.system.execute_action(
                graph=self.current_graph, agent_id=agent_id, actions=actions_list
            )

            error_logs = [
                l for l in logs if "Error" in l or "Failed" in l or "Reject" in l
            ]

            if error_logs:
                return f"REJECT: 执行操作时发生错误: {'; '.join(error_logs)}"

            return "EXECUTED: 操作成功。\n日志:\n" + "\n".join(logs)

        except ValueError as e:
            return f"REJECT: {str(e)}"

        except Exception as e:
            return f"REJECT: GraphTool 处理意图时发生异常: {str(e)}"

    def inject_law_nodes(self, law_contents: List[str]) -> str:
        """Directly inject a list of laws as LAW nodes into the graph.

        This is a privileged operation used by the `LawWorker` to add new
        legal statutes found during its research to the shared debate state.

        Args:
            law_contents: A list of strings, where each string is the full
                content of a legal article.

        Returns:
            A string log summarizing the injection operation.
        """
        if not self.current_graph:
            return "Error: Graph not set."

        executor = GraphExecutor(
            self.current_graph,
            matcher=self.system.dedup_matcher,
            dedup_thresholds=self.system.dedup_thresholds,
        )

        current_step = self.system.step_counter
        count = 0
        logs = []

        for content in law_contents:
            _, log = executor._apply_add_node(
                content=content,
                node_type=NodeType.LAW,
                agent_id="LawWorker",
                current_step=current_step,
            )

            logs.append(log)

            if "✅" in log:
                count += 1

        if count > 0:
            self.current_graph.refresh_context(current_step=current_step)

        return f"已底层注入 {count} 条法条到图谱中。\n日志:\n" + "\n".join(logs)
