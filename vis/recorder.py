import json
import os
import time
from mas.common import ShadowGraph

class SystemRecorder:
    def __init__(self, log_path: str = "system_trace.json"):
        self.log_path = log_path
        self.trace = [] 
        if os.path.exists(log_path): os.remove(log_path)
        
    def log_event(self, step_name: str, 
                  shadow_graph: ShadowGraph = None, 
                  message: str = ""):
        sg_data = None
        
        if shadow_graph:
            try: sg_data = ShadowGraph.to_dict(shadow_graph)
            except Exception as e: print(f"[Recorder] Error serializing graph: {e}")

        snapshot = {
            "timestamp": time.time(),
            "step": step_name,
            "message": message,
            "shadow_graph": sg_data
        }

        self.trace.append(snapshot)
        
    def save(self):
        with open(self.log_path, 'w', encoding='utf-8') as f: json.dump(self.trace, f, indent=2, ensure_ascii=False)
        print(f"[Recorder] Trace saved to {self.log_path}")