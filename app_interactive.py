import streamlit as st
import asyncio
import pandas as pd
from mas.engine import DebateEngine
from mas.config import SystemConfig
from mas.common import NodeType
from vis.app_utils import render_graph, render_global_memory
from data.loader import CaseDataLoader
st.set_page_config(layout="wide", page_title="Legal MAS Console")

st.markdown("""
<style>
    .stChatMessage { padding: 1rem; border-radius: 0.5rem; margin-bottom: 0.5rem; }
    .stChatMessage[data-testid="stChatMessage"] { background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Legal Multi-Agent Debate System")

if 'engine' not in st.session_state:
    st.session_state.engine = DebateEngine(
        config=SystemConfig(),
        judge_config={}
    )
    
    st.session_state.is_setup = False
    
    st.session_state.chat_history = [
        {"role": "system", "content": "Welcome! Please select a case and initialize the system."}
    ]

if 'samples' not in st.session_state:
    loader = CaseDataLoader("data/sampling")
    st.session_state.samples = loader.load_all(limit=20)

engine = st.session_state.engine

with st.sidebar:
    st.header("🎮 Control Center")
    
    if not st.session_state.is_setup:
        st.subheader("📁 Case Selection")
        sample_titles = [f"{s.uid[:8]} - {s.title[:30]}" for s in st.session_state.samples]
        selected_idx = st.selectbox("Choose a case to debate:", range(len(sample_titles)), format_func=lambda i: sample_titles[i])
        selected_case = st.session_state.samples[selected_idx]
        
        with st.expander("📝 Case Preview"):
            st.write(f"**Cause:** {selected_case.cause}")
            st.write(f"**Plaintiff:** {selected_case.plaintiffs}")
            st.write(f"**Defendant:** {selected_case.defendants}")
            st.caption(selected_case.fact_finding[:200] + "...")

        if st.button("🚀 Initialize System", type="primary", use_container_width=True):
            with st.spinner("Setting up engine..."):
                case_dict = selected_case.model_dump()
                asyncio.run(engine.setup(case_data=case_dict, verbose=True))
                st.session_state.is_setup = True
                
                st.session_state.chat_history.append({
                    "role": "system", 
                    "content": f"✅ System Initialized for Case: {selected_case.title}",
                    "details": {"action": "Case Loaded", "cause": selected_case.cause}
                })
                
                st.rerun()
    
    else:
        col_r, col_c = st.columns(2)
        
        with col_r: st.metric("Round", f"{engine.round_idx} / {engine.max_rounds}")
        
        with col_c:
            last_log = engine.get_snapshot().get("last_log", {})
            conv_score = last_log.get("convergence", {}).get("sma", 0.0)
            st.metric("Convergence", f"{conv_score:.4f}")

        st.markdown("---")
        
        if not engine.is_finished:
            turn_name = engine.current_turn.value.capitalize()
            btn_color = "primary" if engine.current_turn.value == "plaintiff" else "secondary"
            
            if st.button(f"▶️ Run {turn_name}", type=btn_color, use_container_width=True):
                with st.spinner(f"Running {turn_name}'s turn..."):
                    asyncio.run(engine.step())
                    log = engine.get_snapshot().get("last_log", {})
                    just_finished = log.get("turn", "system")
                    
                    st.session_state.chat_history.append({
                        "role": just_finished,
                        "content": log.get("action", "Turn Completed"),
                        "details": log
                    })
                    
                    st.rerun()
        
        else: st.success("🏁 Debate Adjudicated")
            
        if st.button("🔄 Reset & Change Case", use_container_width=True):
            if isinstance(engine, DebateEngine): asyncio.run(engine.close_resources())
            
            for key in list(st.session_state.keys()):
                if key != 'samples': del st.session_state[key]
            
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
            if role == "plaintiff": avatar, name = "🔵", "Plaintiff Team"
            elif role == "defendant": avatar, name = "🔴", "Defendant Team"
            else: avatar, name = "🤖", "System / Judge"

            with st.chat_message(name, avatar=avatar):
                st.write(f"**{content}**")
                
                if "dialogue" in details and details["dialogue"]:
                    with st.expander("🔍 Internal Team Dialogue"):
                        for d_msg in details["dialogue"]:
                            st.caption(f"**{d_msg.get('from', '?')}** ➝ **{d_msg.get('to', '?')}**")
                            st.text(d_msg.get("content", ""))
                            st.divider()
                
                if "adjudication_result" in details:
                    adj = details["adjudication_result"]
                    st.success("⚖️ **Verdict Rendered**")
                    with st.expander("📜 Read Judgment Document"): st.markdown(adj.get("document", "No document."))
                    st.write("Claim Adjudication Summary:")
                    st.json(adj.get("claims_status", {}))

    with col_context:
        st.subheader("🕸️ Debate Graph")
        if snapshot.get("shadow_graph"): render_graph(snapshot["shadow_graph"])
        
        if engine.convergence_history:
            st.subheader("📈 Debate Convergence (ΔΦ)")
            df_conv = pd.DataFrame(engine.convergence_history, columns=["Stability Delta"])
            st.line_chart(df_conv)
            st.caption("Lower ΔΦ (SMA) indicates the debate is reaching consensus or exhausting new arguments.")

        st.markdown("---")
        tab_facts, tab_memory, tab_raw = st.tabs(["📂 Evidence & Facts", "🧠 Global Memory", "📝 Raw Logs"])
        
        with tab_facts:
            if snapshot.get("shadow_graph"):
                sg = snapshot["shadow_graph"]
                facts = [d["content"] for n, d in sg.graph.nodes(data=True) if d.get("type") == NodeType.FACT]
                laws = [d["content"] for n, d in sg.graph.nodes(data=True) if d.get("type") == NodeType.LAW]
                col_f, col_l = st.columns(2)
                
                with col_f:
                    st.write(f"**Facts ({len(facts)})**")
                    for f in facts: st.info(f, icon="📄")
                
                with col_l:
                    st.write(f"**Laws ({len(laws)})**")
                    for l in laws: st.warning(l, icon="⚖️")
        
        with tab_memory: render_global_memory(snapshot)
        with tab_raw: st.json(snapshot.get("last_log", {}))

else:
    st.info("👈 Please select a case and click **Initialize System** to begin.")
    
    st.markdown("""
    ### System Architecture Overview
    1. **Debate Engine:** Orchestrates rounds and checks for convergence.
    2. **Shadow Graph:** A dynamic logical graph that evolves as agents make claims.
    3. **Multi-Agent Teams:** 
        - *Controller:* Strategist (Lawyer).
        - *Workers:* Evidence and Law retrieval experts.
    4. **AI Judge:** Adjudicates the final graph state once converged.
    """)