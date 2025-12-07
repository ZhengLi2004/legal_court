import pytest
from mas.common import ShadowGraph, NodeType, EdgeType

def build_complex_case_graph():
    sg = ShadowGraph()
    meta_p1 = {"is_root_claim": True}
    p1 = sg.add_node("要求返还本金100万", NodeType.CLAIM, "P", metadata=meta_p1)
    f1 = sg.add_node("借款合同", NodeType.FACT, "P")
    f2 = sg.add_node("银行转账记录", NodeType.FACT, "P")
    c1 = sg.add_node("借贷关系成立", NodeType.CLAIM, "P")
    sg.add_edge(f1, c1, EdgeType.SUPPORT)
    sg.add_edge(f2, c1, EdgeType.SUPPORT)
    sg.add_edge(c1, p1, EdgeType.SUPPORT)
    c2 = sg.add_node("主张部分款项是赠与", NodeType.CLAIM, "D")
    f3 = sg.add_node("被告的微信聊天记录", NodeType.FACT, "D")
    sg.add_edge(f3, c2, EdgeType.SUPPORT)
    sg.add_edge(c2, p1, EdgeType.CONFLICT)
    meta_p2 = {"is_root_claim": True}
    p2 = sg.add_node("要求支付利息5万", NodeType.CLAIM, "P", metadata=meta_p2)
    f4 = sg.add_node("合同约定了年化24%利率", NodeType.FACT, "P")
    sg.add_edge(f4, p2, EdgeType.SUPPORT)
    f5 = sg.add_node("资金流水显示用于赌博", NodeType.FACT, "P")
    sg.add_edge(f5, p1, EdgeType.SUPPORT)
    return sg

def test_serialization_quality():
    sg = build_complex_case_graph()
    text = sg.to_recursive_text()
    print(f"\n--- 智能序列化输出 ---\n{text}\n--------------------")
    assert '议题 1: 关于 “要求返还本金100万...”' in text
    assert '议题 2: 关于 “要求支付利息5万...”' in text
    idx_issue1 = text.find('议题 1: 关于 “要求返还本金100万...”')
    idx_issue2 = text.find('议题 2: 关于 “要求支付利息5万...”')
    assert idx_issue1 < idx_issue2
    issue1_block = text[idx_issue1:idx_issue2]
    idx_p1 = issue1_block.find("要求返还本金100万")
    idx_c1 = issue1_block.find("借贷关系成立")
    idx_c2 = issue1_block.find("主张部分款项是赠与")
    assert idx_p1 != -1 and idx_c1 != -1 and idx_c2 != -1
    assert idx_p1 < idx_c1
    assert idx_p1 < idx_c2
    issue2_block = text[idx_issue2:]
    idx_p2 = issue2_block.find("要求支付利息5万")
    idx_f4 = issue2_block.find("合同约定了年化24%利率")
    assert idx_p2 != -1 and idx_f4 != -1
    assert idx_p2 < idx_f4

if __name__ == "__main__": pytest.main(["-s", "test/test_serialization.py"])