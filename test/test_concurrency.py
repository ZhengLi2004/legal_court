import pytest
import os
import json
from multiprocessing import Process
from mas.insights_manager import InsightsManager
from mas.config import SystemConfig
class MockLLM: pass

class MockEmbeddingFunc:
    def embed_query(self, text: str) -> list: return [0.1] * 10

class MockMatcher:
    def __init__(self): self.embedding_func = MockEmbeddingFunc()

def worker_function(persist_dir: str, worker_id: int, num_writes: int):
    mgr = InsightsManager(persist_dir, llm=MockLLM(), matcher=MockMatcher(), config=SystemConfig())

    for i in range(num_writes):
        insight_content = f"Insight from worker {worker_id}, write #{i}"
        mgr.insights = mgr._load_insights()
        mgr.insights.append({"content": insight_content})
        insights_as_dicts = []
        
        for item in mgr.insights:
            if isinstance(item, dict): insights_as_dicts.append(item)
            else: insights_as_dicts.append(item.__dict__)

        mgr.insights = insights_as_dicts
        from mas.utils import file_lock
        lock_file = mgr.file_path + ".lock"

        try:
            with file_lock(lock_file):
                current_data = []

                if os.path.exists(mgr.file_path):
                    with open(mgr.file_path, 'r', encoding='utf-8') as f:
                        try: current_data = json.load(f)
                        except json.JSONDecodeError: pass

                current_data.append({"content": insight_content})
                with open(mgr.file_path, 'w', encoding='utf-8') as f: json.dump(current_data, f, indent=2, ensure_ascii=False)
            
        except TimeoutError: print(f"Worker {worker_id} failed to acquire lock.")

def test_concurrent_writers(tmp_path):
    persist_dir = str(tmp_path)
    num_processes = 5
    writes_per_process = 10
    total_writes = num_processes * writes_per_process
    insights_file = os.path.join(persist_dir, "legal_insights.json")
    if os.path.exists(insights_file): os.remove(insights_file)
    processes = []
    
    for i in range(num_processes):
        p = Process(target=worker_function, args=(persist_dir, i, writes_per_process))
        processes.append(p)
        p.start()

    for p in processes: p.join()
    assert os.path.exists(insights_file)
    with open(insights_file, 'r', encoding='utf-8') as f: final_data = json.load(f)
    assert len(final_data) == total_writes, "Lock failed: concurrent writes were lost."

if __name__ == "__main__": pytest.main(["-v", "test/test_concurrency.py"])