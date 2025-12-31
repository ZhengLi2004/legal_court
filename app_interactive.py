import asyncio

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
        node_type = data.get("type")

        if node_type == NodeType.FACT:
            stats["facts"] += 1

        elif node_type == NodeType.LAW:
            stats["laws"] += 1

        elif node_type == NodeType.CLAIM:
            stats["claims"] += 1

    for _, _, data in graph.edges(data=True):
        edge_type = data.get("type")

        if edge_type == EdgeType.SUPPORT:
            stats["support_edges"] += 1

        elif edge_type == EdgeType.CONFLICT:
            stats["conflict_edges"] += 1

    return stats


def render_verdict_summary(adjudication_result):
    st.success("⚖️ **Verdict Rendered**")

    with st.expander("📜 Read Full Judgment Document"):
        st.markdown(adjudication_result.get("document", "No document."))

    st.write("**Claim Adjudication Summary:**")
    claims_status = adjudication_result.get("claims_status", {})

    if not claims_status:
        st.info("No root claims were adjudicated.")
        return

    for claim_id, status in claims_status.items():
        col1, col2 = st.columns([2, 1])

        with col1:
            st.caption(f"ID: {claim_id}")

        with col2:
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

    for _, msg in enumerate(memory_list):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if role == "System":
            avatar = "⚙️"
            bg_color = "#f0f2f6"
        elif "Controller" in role or role == "user":
            avatar = "🧠"
            bg_color = "#e8f0fe"

        elif "Worker" in role:
            avatar = "👷"
            bg_color = "#fff8e1"

        else:
            avatar = "❓"
            bg_color = "#ffffff"

        with st.container():
            c1, c2 = st.columns([1, 10])

            with c1:
                st.markdown(f"**{avatar} {role}**")

            with c2:
                if "=== 🕵️ 本轮调查综述" in content:
                    st.success(content)

                elif "WORKERS_COMPLETED" in content:
                    with st.expander(
                        "📶 Signal: WORKERS_COMPLETED (Click to view payload)"
                    ):
                        st.code(content, language="json")
                else:
                    st.info(content)

        st.divider()


st.set_page_config(layout="wide", page_title="Legal MAS Console")

st.markdown(
    """
<style>
    .stChatMessage { padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem; }
    .stChatMessage[data-testid="stChatMessage"] { background-color: #f0f2f6; }
    div[data-testid="stMetric"] { background-color: #f0f2f6; border-radius: 0.5rem; padding: 10px; }
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
            "Choose a case to debate:",
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
            st.metric("Convergence (SMA)", f"{conv_score:.4f}")

        st.markdown("---")

        if not engine.is_finished:
            turn_name = engine.current_turn.value.capitalize()

            btn_color = (
                "primary" if engine.current_turn.value == "plaintiff" else "secondary"
            )

            if st.button(
                f"▶️ Run {turn_name}'s Turn", type=btn_color, use_container_width=True
            ):
                with st.spinner(f"Running {turn_name}'s turn..."):
                    asyncio.run(engine.step())
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

        if st.button("🔄 Reset & Change Case", use_container_width=True):
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
            content = msg["content"]
            details = msg.get("details", {})

            if role == "plaintiff":
                avatar, name = "🔵", "Plaintiff Team"

            elif role == "defendant":
                avatar, name = "🔴", "Defendant Team"

            else:
                avatar, name = "🤖", "System / Judge"

            with st.chat_message(name, avatar=avatar):
                if role in ["plaintiff", "defendant"]:
                    st.markdown(f"**Action:** {content}")

                    if "dialogue" in details and details["dialogue"]:
                        with st.expander(
                            "🕵️ Worker Execution Logs (Parallel)", expanded=False
                        ):
                            for d_msg in details["dialogue"]:
                                sender = d_msg.get("from", "?")
                                receiver = d_msg.get("to", "?")
                                st.caption(f"`{sender}` ➝ `{receiver}`")
                                txt = d_msg.get("content", "")

                                if "🔎" in txt or "⚖️" in txt or "🧠" in txt:
                                    st.markdown(txt)

                                else:
                                    st.code(
                                        txt, language="json" if "{" in txt else "text"
                                    )

                elif "adjudication_result" in details:
                    render_verdict_summary(details["adjudication_result"])

                else:
                    st.markdown(content)

    with col_context:
        tab_graph, tab_memory, tab_agent_ctx, tab_raw = st.tabs(
            ["🕸️ Graph", "🧠 Global Memory", "🤖 Agent Context", "📝 Logs"]
        )

        with tab_graph:
            if snapshot.get("shadow_graph"):
                render_graph(snapshot["shadow_graph"])
                stats = get_graph_stats(snapshot["shadow_graph"].graph)
                c1, c2, c3 = st.columns(3)
                c1.metric("Facts", stats.get("facts", 0))
                c2.metric("Laws", stats.get("laws", 0))
                c3.metric("Claims", stats.get("claims", 0))

        with tab_memory:
            render_global_memory(snapshot)

        with tab_agent_ctx:
            st.caption("Real-time internal memory of the active agents.")
            last_turn = snapshot.get("last_log", {}).get("turn", "plaintiff")

            selected_agent = st.radio(
                "Select Agent View:",
                ["plaintiff", "defendant"],
                index=0 if last_turn == "plaintiff" else 1,
                horizontal=True,
            )

            memories = snapshot.get("agent_memories", {}).get(selected_agent, [])

            if memories:
                st.markdown(
                    f"### {selected_agent.capitalize()} Controller Memory ({len(memories)} items)"
                )

                render_agent_memory(memories)

            else:
                st.info("No memory initialized for this agent.")

        with tab_raw:
            st.json(snapshot.get("last_log", {}))

else:
    st.info("👈 Please select a case and click **Initialize System** to begin.")
