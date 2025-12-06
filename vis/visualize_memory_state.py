from mas.legal_system import LegalSystem
from vis.static_viz import GMemoryVisualizer
import os

def snapshot_global_state(system: LegalSystem, round_id: int, output_dir: str = "./viz_output/global_evolution"):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    viz = GMemoryVisualizer(output_dir=output_dir)
    q_filename = f"round_{round_id:02d}_query_graph.png"
    viz.draw_query_graph(system.memory.task_layer, filename=q_filename)
    i_filename = f"round_{round_id:02d}_insight_graph.png"
    viz.draw_insight_graph(system.insights, filename=i_filename)
    print(f"[Snapshot] Saved global state for Round {round_id}")