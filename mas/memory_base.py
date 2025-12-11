import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List
from .utils import EmbeddingFunc
from .common import LegalMessage
from .config import SystemConfig

@dataclass
class MASMemoryBase(ABC):
    persist_dir: str
    embedding_model_path: str = None

    def __post_init__(self):
        if self.embedding_model_path is None: self.embedding_model_path = SystemConfig().path.embedding_model_path
        self.embedding_func = EmbeddingFunc(model_path=self.embedding_model_path)
        if not os.path.exists(self.persist_dir): os.makedirs(self.persist_dir)

    @abstractmethod
    def add_memory(self, message: LegalMessage) -> None: pass
    @abstractmethod
    def retrieve_memory(self, query_context: str, top_k: int = 3) -> Tuple[List[LegalMessage], List[str]]: pass
    # 清空记忆（测试用途）
    def reset(self):
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            os.makedirs(self.persist_dir)