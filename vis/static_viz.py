import networkx as nx
import matplotlib.pyplot as plt
import os
from mas.common import ShadowGraph, NodeStatus
from mas.task_layer import TaskLayer
from mas.insights_manager import InsightsManager

class GMemoryVisualizer:
    def __init__(self, output_dir: str = "./viz_output"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
    # 1. Interaction (Shadow) Graph
    def draw_shadow_graph(self, shadow_graph: ShadowGraph, filename: str = "shadow_graph.png", title: str = ""):
        G = shadow_graph.graph
        
        if G.number_of_nodes() == 0: 
            print("[Viz] ShadowGraph is empty.")
            return

        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=1.5, seed=42)
        node_colors = []
        edgecolors = []
        labels = {}
        
        for n, d in G.nodes(data=True):
            n_type = d.get('type')
            if hasattr(n_type, 'value'): n_type = n_type.value
            c = 'gray'
            if str(n_type) == 'FACT': c = '#87CEFA'
            elif str(n_type) == 'LAW': c = '#FFD700'
            elif str(n_type) == 'CLAIM': c = '#90EE90'
            node_colors.append(c)
            status = d.get('status', NodeStatus.HYPOTHETICAL)
            if str(status) == 'VALIDATED': edgecolors.append('green')
            elif str(status) == 'DEFEATED': edgecolors.append('red')
            else: edgecolors.append('gray') # Hypothetical
            content = d.get('content', '')
            labels[n] = f"{str(n_type)}\n{content[:8]}..." if len(content)>8 else f"{str(n_type)}\n{content}"

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, edgecolors=edgecolors, linewidths=2, node_size=2000)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)
        edge_colors_list = []
        styles = []

        for _, _, d in G.edges(data=True):
            e_type = d.get('type')
            
            if str(e_type) == 'SUPPORT':
                edge_colors_list.append('green')
                styles.append('solid')
            
            else:
                edge_colors_list.append('red')
                styles.append('dashed')
                
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors_list, style=styles, arrowstyle='-|>', arrowsize=20, width=2)
        plt.title(title or "Interaction (Shadow) Graph")
        plt.axis('off')
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Viz] Saved ShadowGraph to {save_path}")
    # 2. Query Graph (Topology)
    def draw_query_graph(self, task_layer: TaskLayer, filename: str = "query_graph.png"):
        G = task_layer.graph
        if G.number_of_nodes() == 0: return
        plt.figure(figsize=(10, 8))
        try: pos = nx.kamada_kawai_layout(G)
        except: pos = nx.spring_layout(G, k=0.5)
        nx.draw_networkx_nodes(G, pos, node_color='#D8BFD8', node_size=800, alpha=0.8)
        labels = {n: str(n)[-6:] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        weights = [d.get('weight', 0.5) for _, _, d in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, width=[w*2 for w in weights], edge_color='gray', alpha=0.5)
        plt.title("Query Graph (Topology)")
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close()
    # 3. Insight Graph (Strategy Map)
    def draw_insight_graph(self, insights_manager: InsightsManager, filename: str = "insight_graph.png"):
        insights = insights_manager.insights
        if not insights: return
        G = nx.Graph()
        for i, inst in enumerate(insights): G.add_node(i, content=inst.content, score=inst.score)
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=3.0)
        sizes = [d['score'] * 300 + 500 for _, d in G.nodes(data=True)]
        nx.draw_networkx_nodes(G, pos, node_color='#FFA07A', node_size=sizes, alpha=0.9)
        labels = {n: d['content'][:15] + ".." for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        plt.title(f"Insight Graph ({len(insights)} Strategies)")
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, filename), dpi=150)
        plt.close()