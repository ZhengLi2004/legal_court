import os
import yaml
import json
import random
import math
import numpy as np
import contextlib
import os
import time
from dataclasses import dataclass
from typing import Union, Any, List
# 锁
@contextlib.contextmanager
def simple_file_lock(lock_path: str, timeout: int = 5):
    start_time = time.time()
    
    while os.path.exists(lock_path):
        if time.time() - start_time > timeout: raise TimeoutError(f"Could not acquire lock: {lock_path}")
        time.sleep(0.1)

    with open(lock_path, 'w') as f: f.write('LOCKED')
    try: yield
    
    finally:
        if os.path.exists(lock_path): os.remove(lock_path)
# 配置加载
def load_config(config_path: str = "configs/configs.yaml"):
    if not os.path.exists(config_path): return {}
    with open(config_path, "r", encoding="utf-8") as file: return yaml.safe_load(file)

def load_json(file_name: str) -> Union[list, dict]:
    if not os.path.exists(file_name): return None
    with open(file_name, encoding="utf-8") as f: return json.load(f)
# 列表分块，最长长度为 k
def random_divide_list(lst: list[Any], k: int) -> list[list]:
    if len(lst) == 0: return []
    lst_copy = list(lst)
    random.shuffle(lst_copy)
    if len(lst_copy) <= k: return [lst_copy]

    else:
        num_chunks = math.ceil(len(lst_copy) / k)
        chunk_size = math.ceil(len(lst_copy) / num_chunks)
        return [lst_copy[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
# 计算余弦相似度
def cosine_similarity(vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]) -> float:
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))
# 缓存模型
_EMBEDDING_MODEL_CACHE = {}

@dataclass
class EmbeddingFunc:
    model_path: str = "./bge-m3"
    # 延迟导入
    def __post_init__(self):
        from chromadb.utils import embedding_functions

        if self.model_path not in _EMBEDDING_MODEL_CACHE:
            print(f"[EmbeddingFunc] Loading model from: {self.model_path}")
            if not os.path.exists(self.model_path): raise FileNotFoundError(f"Embedding model not found at {self.model_path}")
                
            _EMBEDDING_MODEL_CACHE[self.model_path] = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.model_path
            )

        self.func = _EMBEDDING_MODEL_CACHE[self.model_path]

    def embed_documents(self, texts: list[str]) -> list[list]: return self.func(texts)
    def embed_query(self, query: str) -> list: return self.func([query])[0]