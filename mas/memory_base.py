"""Defines the abstract base class for a memory system in the MAS.

This module provides `MASMemoryBase`, an ABC that outlines the essential
interface for any long-term memory component in the system. It ensures that
all memory implementations will have common methods for adding and retrieving
memories, as well as for initialization and reset.
"""

import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

from .common import LegalMessage
from .config import SystemConfig
from .utils import EmbeddingFunc


@dataclass
class MASMemoryBase(ABC):
    """Abstract base class for a multi-agent system's long-term memory.

    This class provides a common structure for memory systems, handling
    the initialization of the embedding function and the persistent directory.
    Concrete implementations must provide the logic for `add_memory` and
    `retrieve_memory`.

    Attributes:
        persist_dir: The file system path where memory data is stored.
        embedding_model_path: The path to the sentence transformer model.
        embedding_func: An instance of `EmbeddingFunc` for vectorizing text.
    """

    persist_dir: str
    embedding_model_path: str = None

    def __post_init__(self):
        """Initialize the embedding function and ensures the persist directory exists."""
        if self.embedding_model_path is None:
            self.embedding_model_path = SystemConfig().path.embedding_model_path

        self.embedding_func = EmbeddingFunc(model_path=self.embedding_model_path)

        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)

    @abstractmethod
    def add_memory(self, message: LegalMessage) -> None:
        """Add a new memory (a completed case) to the storage.

        Args:
            message: A `LegalMessage` object representing the case to be stored.
        """
        pass

    @abstractmethod
    def retrieve_memory(
        self, query_context: str, top_k: int = 3
    ) -> Tuple[List[LegalMessage], List[str]]:
        """Retrieve relevant memories based on a query.

        Args:
            query_context: A string describing the current situation or case.
            top_k: The maximum number of memories to retrieve.

        Returns:
            A tuple containing a list of retrieved `LegalMessage` objects and a
            list of any associated debug strings or metadata.
        """
        pass

    def reset(self):
        """Delete all stored memories and resets the memory to a clean state."""
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            os.makedirs(self.persist_dir)
