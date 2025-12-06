import json
import os
import numpy as np
import imageio.v2 as imageio
from .static_viz import GMemoryVisualizer
from mas.common import ShadowGraph, json_graph
from PIL import Image

def generate_dynamic_gif(trace_path: str, output_gif: str = "evolution.gif", duration: float = 1.5):
    if not os.path.exists(trace_path):
        print(f"Trace file {trace_path} not found.")
        return

    with open(trace_path, 'r', encoding='utf-8') as f: trace = json.load(f)
    temp_dir = "./temp_frames"
    viz = GMemoryVisualizer(output_dir=temp_dir)
    images = []
    print(f"[DynamicViz] Generating frames for {len(trace)} steps...")
    target_size = (1200, 1000)

    for i, step in enumerate(trace):
        if step.get('shadow_graph'):
            try:
                sg_data = step['shadow_graph']
                sg = ShadowGraph()
                graph_dict = sg_data["graph_data"]
                sg.id_alias = sg_data.get("id_alias", {})
                sg.graph = json_graph.node_link_graph(graph_dict)
                filename = f"frame_{i:03d}.png"
                title = f"Step {i}: {step['step']}\n{step['message']}"
                viz.draw_shadow_graph(sg, filename=filename, title=title)
                img_path = os.path.join(temp_dir, filename)
                
                if os.path.exists(img_path):
                    with Image.open(img_path) as img:
                        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                        images.append(np.array(img_resized))
            
            except KeyError as e: print(f"KeyError processing frame {i}: {e}. Data: {sg_data}")
            except Exception as e: print(f"Error processing frame {i}: {e}")
            
    if images:
        imageio.mimsave(output_gif, images, duration=duration)
        print(f"✅ GIF saved to {output_gif}")
    
    else: print("❌ No images generated.")