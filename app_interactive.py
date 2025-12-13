import streamlit as st
import asyncio
import json
from mas.engine import DebateEngine
from mas.config import SystemConfig
from vis.app_utils import render_graph, render_global_memory, JUDGE_CONFIG
st.set_page_config(layout="wide", page_title="Legal MAS Debugger")
st.title("⚖️ Legal Multi-Agent Debate Console")

if 'engine' not in st.session_state:
    st.session_state.engine = DebateEngine(
        config=SystemConfig(),
        judge_config=JUDGE_CONFIG
    )

    st.session_state.is_setup = False
    st.session_state.log_messages = ["Welcome! Please initialize the system."]

engine = st.session_state.engine

with st.sidebar:
    st.header("⚙️ Control Panel")
    
    if not st.session_state.is_setup:
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Setting up engine..."):
                asyncio.run(engine.setup("data/sampling/cleaned_samples.jsonl", verbose=True))
                st.session_state.is_setup = True
                st.session_state.log_messages.append({"action": "✅ System Initialized."})
                st.rerun()
    
    else:
        if not engine.is_finished:
            turn_name = engine.current_turn.value.capitalize()
            
            if st.button(f"▶️ Run {turn_name}'s Turn", type="primary"):
                with st.spinner(f"Running {turn_name}'s turn..."):
                    asyncio.run(engine.step())
                    log = engine.get_snapshot().get("last_log", {})
                    st.session_state.log_messages.append(log)
                    st.rerun()
        
        else: st.success(f"Debate Finished! Winner: {engine.winner}")
            
        if st.button("🔄 Reset & New Case"):
            if isinstance(engine, DebateEngine): asyncio.run(engine.close_resources())
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()

    st.markdown("---")
    st.header("🧠 Global Memory State")
    
    if st.session_state.is_setup:
        snapshot = engine.get_snapshot()
        with st.expander("Show Insight & Topology Graphs"): render_global_memory(snapshot)

if st.session_state.is_setup:
    snapshot = engine.get_snapshot()
    col_graph, col_logs = st.columns([2, 1])

    with col_graph:
        st.subheader("🕸️ Debate Graph")
        if snapshot.get("shadow_graph"): render_graph(snapshot["shadow_graph"])

    with col_logs:
        st.subheader("📜 Turn History")

        for i, log_item in enumerate(reversed(st.session_state.log_messages)):
            if isinstance(log_item, dict):
                turn_info = f"Round {log_item.get('round', '?')} - {log_item.get('turn', '?').capitalize()}"
                
                with st.expander(f"Step {len(st.session_state.log_messages) - i}: {turn_info}", expanded=(i == 0)):
                    st.markdown(f"**Action Summary:**")
                    st.info(log_item.get('action', 'N/A'))
                    dialogue = log_item.get("dialogue", [])
                    
                    if dialogue:
                        st.markdown("**Internal Dialogue:**")
                        
                        for msg in dialogue:
                            sender = msg.get("from", "?")
                            receiver = msg.get("to", "?")
                            content = msg.get("content", "")
                            st.markdown(f"*{sender} ➡️ {receiver}*")
                            
                            try:
                                data = json.loads(content)
                                st.json(data)
                            
                            except:st.text(content)
            
            else: st.info(log_item)

else: st.info("Please initialize the system using the button on the left.")