from typing import List

from .common import LegalMessage, NodeType, ShadowGraph
from .config import SystemConfig
from .semantic_matcher import SemanticMatcher


class GraphProjector:
    def __init__(self, matcher: SemanticMatcher, config: SystemConfig = None):
        self.matcher = matcher
        self.cfg = config or SystemConfig()

    def retrieve_historical_context(
        self,
        current_graph: ShadowGraph,
        focus_node_ids: List[str],
        history_messages: List[LegalMessage],
    ) -> str:
        if not focus_node_ids:
            return "（无选定锚点，无法进行映射）"

        anchors_content = []

        for nid in focus_node_ids:
            if current_graph.graph.has_node(nid):
                anchors_content.append(
                    current_graph.graph.nodes[nid].get("content", "")
                )

        if not anchors_content:
            return "（锚点内容为空）"

        context_texts = []

        for msg in history_messages:
            hist_text = self._extract_context_from_single_history(
                msg.shadow_graph, anchors_content, msg.case_id
            )

            if hist_text:
                context_texts.append(
                    f"\n>>> 历史案例 [{msg.case_id[:8]}] 参考:\n{hist_text}"
                )

        if not context_texts:
            return "（在历史案例中未找到与当前锚点足够相似的对应论点）"

        return "\n".join(context_texts)

    def _extract_context_from_single_history(
        self,
        history_graph: ShadowGraph,
        target_contents: List[str],
        case_id: str,
    ) -> str:
        source_candidates = [
            (nid, data["content"]) for nid, data in history_graph.graph.nodes(data=True)
        ]

        matched_history_ids = set()

        for tgt_content in target_contents:
            hist_id = self.matcher.find_match(tgt_content, source_candidates)

            if hist_id:
                matched_history_ids.add(hist_id)

        if not matched_history_ids:
            return ""

        nodes_to_serialize = set(matched_history_ids)

        for hist_id in matched_history_ids:
            predecessors = list(history_graph.graph.predecessors(hist_id))
            successors = list(history_graph.graph.successors(hist_id))
            neighbors = predecessors + successors
            count = 0

            for nid in neighbors:
                nodes_to_serialize.add(nid)
                count += 1

        subgraph = history_graph.get_subgraph(list(nodes_to_serialize))
        return subgraph.to_recursive_text()
