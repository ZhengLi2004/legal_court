import asyncio
import datetime

import pandas as pd
import streamlit as st

from data.loader import CaseDataLoader
from mas.common import EdgeType, NodeType
from mas.config import SystemConfig
from mas.engine import DebateEngine
from vis.app_utils import render_global_memory, render_graph


def get_graph_stats(graph):
    if not graph:
        return {}

    stats = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "facts": 0,
        "laws": 0,
        "claims": 0,
        "support_edges": 0,
        "conflict_edges": 0,
    }

    for _, data in graph.nodes(data=True):
        t = data.get("type")

        if t == NodeType.FACT:
            stats["facts"] += 1

        elif t == NodeType.LAW:
            stats["laws"] += 1

        elif t == NodeType.CLAIM:
            stats["claims"] += 1

    for _, _, data in graph.edges(data=True):
        t = data.get("type")

        if t == EdgeType.SUPPORT:
            stats["support_edges"] += 1

        elif t == EdgeType.CONFLICT:
            stats["conflict_edges"] += 1

    return stats


def render_verdict_summary(adjudication_result):
    st.success("⚖️ **Verdict Rendered**")
    document_content = adjudication_result.get("document", "No document.")

    paper_style = """
    <div style="background-color: #f9f9f9; padding: 40px; border: 1px solid #ddd; border-radius: 5px; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); font-family: 'Times New Roman', serif; margin-bottom: 20px;">
        <h2 style="text-align: center; color: #333; margin-bottom: 5px;">民 事 判 决 书</h2>
        <p style="text-align: center; color: #666; font-size: 0.9em;">(AI Adjudication Draft)</p>
        <hr style="border-top: 2px solid #333; margin-top: 10px; margin-bottom: 20px;">
        <div style="font-size: 16px; line-height: 1.8; color: #222; text-align: justify;">{content}</div>
        <br><br>
        <div style="text-align: right; margin-top: 30px;"><p><strong>本案 AI 审判员</strong></p><p>{date}</p></div>
    </div>
    """

    formatted_html = paper_style.format(
        content=document_content.replace("\n", "<br>"),
        date=datetime.date.today().strftime("%Y年%m月%d日"),
    )

    with st.expander("📜 Read Full Judgment Document", expanded=False):
        st.markdown(formatted_html, unsafe_allow_html=True)

    st.write("**Claim Adjudication Summary:**")
    claims_status = adjudication_result.get("claims_status", {})

    if not claims_status:
        st.info("No root claims were adjudicated.")
        return

    for claim_id, status in claims_status.items():
        c1, c2 = st.columns([2, 1])

        with c1:
            st.caption(f"ID: {claim_id}")

        with c2:
            if status == "VALIDATED":
                st.success(f"✔️ {status}", icon="✔️")

            elif status == "DEFEATED":
                st.error(f"❌ {status}", icon="❌")

            else:
                st.warning(f"➖ {status}", icon="➖")

        st.divider()


def render_agent_memory(memory_list):
    if not memory_list:
        st.info("Memory is empty.")
        return

    for msg in memory_list:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "System":
            avatar = "⚙️"

        elif "Controller" in role:
            avatar = "🧠"

        elif "Worker" in role:
            avatar = "👷"

        else:
            avatar = "❓"

        with st.container():
            c1, c2 = st.columns([1, 15])

            with c1:
                st.markdown(f"**{avatar}**")

            with c2:
                st.caption(f"**{role}**")

                if "=== 🕵️ 本轮调查综述" in content:
                    st.success(content)

                elif "Worker investigation finished" in content:
                    st.caption(f"*{content}*")

                else:
                    st.text(content)

        st.divider()


st.set_page_config(layout="wide", page_title="Legal MAS Console")

st.markdown(
    """
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        border: 1px solid rgba(0,0,0,0.05);
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }

    .stChatMessage[data-testid="stChatMessage"] {
        background-color: #ffffff; /* 默认白色 */
    }
    
    .chat-bubble-plaintiff {
        background-color: #e3f2fd !important;
        border-left: 5px solid #2196f3 !important;
    }

    .chat-bubble-defendant {
        background-color: #ffebee !important;
        border-left: 5px solid #f44336 !important;
    }

    .chat-bubble-system {
        background-color: #f5f5f5 !important;
        border-left: 5px solid #9e9e9e !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("⚖️ Legal Multi-Agent Debate System")

if "engine" not in st.session_state:
    st.session_state.engine = DebateEngine(config=SystemConfig(), judge_config={})
    st.session_state.is_setup = False

    st.session_state.chat_history = [
        {
            "role": "system",
            "content": "Welcome! Please select a case and initialize the system.",
        }
    ]

if "samples" not in st.session_state:
    loader = CaseDataLoader("data/sampling")
    st.session_state.samples = loader.load_all(limit=20)

engine = st.session_state.engine

with st.sidebar:
    st.header("🎮 Control Center")

    if not st.session_state.is_setup:
        st.subheader("📁 Case Selection")

        sample_titles = [
            f"{s.uid[:8]} - {s.title[:30]}" for s in st.session_state.samples
        ]

        selected_idx = st.selectbox(
            "Choose a case:",
            range(len(sample_titles)),
            format_func=lambda i: sample_titles[i],
        )

        selected_case = st.session_state.samples[selected_idx]

        with st.expander("📝 Case Preview"):
            st.write(f"**Cause:** {selected_case.cause}")
            st.caption(selected_case.fact_finding[:200] + "...")

        if st.button("🚀 Initialize System", type="primary", use_container_width=True):
            with st.spinner("Setting up engine..."):
                case_dict = selected_case.model_dump()
                asyncio.run(engine.setup(case_data=case_dict, verbose=True))
                st.session_state.is_setup = True
                initial_stats = get_graph_stats(engine.graph.graph)

                init_content = (
                    f"✅ System Initialized for Case: {selected_case.title}\n\n"
                    f"- **{initial_stats.get('facts', 0)}** objective facts injected.\n"
                    f"- **{initial_stats.get('claims', 0)}** root claims established."
                )

                st.session_state.chat_history.append(
                    {
                        "role": "system",
                        "content": init_content,
                        "details": {"action": "Case Loaded"},
                    }
                )

                st.rerun()

    else:
        col_r, col_c = st.columns(2)

        with col_r:
            st.metric("Round", f"{engine.round_idx} / {engine.max_rounds}")

        last_log = engine.get_snapshot().get("last_log", {})
        conv_score = last_log.get("convergence", {}).get("sma", 0.0)

        with col_c:
            st.metric("Convergence", f"{conv_score:.4f}")

        st.markdown("---")

        if not engine.is_finished:
            turn_name = engine.current_turn.value.capitalize()

            btn_color = (
                "primary" if engine.current_turn.value == "plaintiff" else "secondary"
            )

            if st.button(
                f"▶️ Run {turn_name}'s Turn", type=btn_color, use_container_width=True
            ):
                with st.status(
                    f"Processing {turn_name}'s Turn...", expanded=True
                ) as status:
                    st.write(
                        "🧠 Controller is assessing needs & dispatching workers..."
                    )

                    asyncio.run(engine.step())
                    st.write("📝 Synthesizing Narrative & Updating Graph...")

                    status.update(
                        label=f"✅ {turn_name}'s Turn Completed",
                        state="complete",
                        expanded=False,
                    )

                    log = engine.get_snapshot().get("last_log", {})
                    just_finished = log.get("turn", "system")

                    st.session_state.chat_history.append(
                        {
                            "role": just_finished,
                            "content": log.get("action", "Turn Completed"),
                            "details": log,
                        }
                    )

                    st.rerun()

        else:
            st.success("🏁 Debate Adjudicated")

        if st.button("🔄 Reset", use_container_width=True):
            if isinstance(engine, DebateEngine):
                asyncio.run(engine.close_resources())

            for key in list(st.session_state.keys()):
                if key != "samples":
                    del st.session_state[key]

            st.rerun()

if st.session_state.is_setup:
    snapshot = engine.get_snapshot()
    col_chat, col_context = st.columns([5, 4])

    with col_chat:
        st.subheader("💬 Debate Stream")

        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]  # 默认内容 (Action Summary)
            details = msg.get("details", {})
            narrative = details.get("narrative", "")  # 叙事内容

            if role == "plaintiff":
                avatar, name = "🔵", "原告代理人 (Plaintiff)"
                css_class = "chat-bubble-plaintiff"

            elif role == "defendant":
                avatar, name = "🔴", "被告代理人 (Defendant)"
                css_class = "chat-bubble-defendant"

            else:
                avatar, name = "🤖", "System / Judge"
                css_class = "chat-bubble-system"

            with st.chat_message(name, avatar=avatar):
                if narrative:
                    if role == "plaintiff":
                        st.info(narrative, icon="🗣️")

                    elif role == "defendant":
                        st.error(narrative, icon="🗣️")  # 用 error 红色框代表被告

                    else:
                        st.markdown(narrative)

                elif "adjudication_result" in details:
                    render_verdict_summary(details["adjudication_result"])

                else:
                    st.markdown(content)

                if role in ["plaintiff", "defendant"]:
                    with st.expander(
                        "🛠️ Technical Logs (Worker & Action)", expanded=False
                    ):
                        if "dialogue" in details and details["dialogue"]:
                            st.caption("Worker Execution Logs:")

                            for d_msg in details["dialogue"]:
                                sender = d_msg.get("from", "").split("_")[-1]

                                if "Worker" in sender:
                                    txt = d_msg.get("content", "")
                                    st.text(f"[{sender}]: {txt[:50]}...")

                        st.caption("Graph Action:")
                        st.code(content, language="text")

    with col_context:
        tab_graph, tab_memory, tab_agent_ctx, tab_raw = st.tabs(
            ["🕸️ Graph", "🧠 Memory", "🤖 Context", "📝 Logs"]
        )

        with tab_graph:
            st.caption("Legend: 🔵Fact 🟡Law 🟢Claim | 🟩Support 🟥Conflict")

            if snapshot.get("shadow_graph"):
                render_graph(snapshot["shadow_graph"])

            if engine.convergence_history:
                st.divider()
                st.caption("Debate Convergence (SMA)")
                df_conv = pd.DataFrame(engine.convergence_history, columns=["ΔΦ"])
                st.line_chart(df_conv, height=150)

        with tab_memory:
            render_global_memory(snapshot)

        with tab_agent_ctx:
            st.subheader("Active Agent Memory")
            last_turn = snapshot.get("last_log", {}).get("turn", "plaintiff")

            selected_agent = st.radio(
                "Select Agent:",
                ["plaintiff", "defendant"],
                index=0 if last_turn == "plaintiff" else 1,
                horizontal=True,
            )

            memories = snapshot.get("agent_memories", {}).get(selected_agent, [])

            if memories:
                render_agent_memory(memories)

            else:
                st.info("No memory initialized.")

        with tab_raw:
            st.json(snapshot.get("last_log", {}))

else:
    st.info("👈 Please select a case and click **Initialize System** to begin.")

    st.markdown("""
    ### 🏛️ Legal MAS System
    An advanced Neuro-Symbolic debate system.
    
    **Workflow:**
    1.  **Parallel Investigation**: Fact/Law/Strategy workers run concurrently.
    2.  **Narrative Construction**: Programmatic logic + LLM polishing creates readable transcripts.
    3.  **Graph Evolution**: Explicit logic actions update the Shadow Graph.
    4.  **Narrative Adjudication**: Judge reads the full transcript to render a verdict.
    """)
