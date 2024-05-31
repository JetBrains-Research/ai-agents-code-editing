from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from code_editing.configs.agents.embeddings_config import EmbeddingsConfig
from code_editing.configs.utils import CE_CLASSES_ROOT_PKG


@dataclass
class RetrievalConfig:
    _target_: str = MISSING
    splitter: Any = MISSING


@dataclass
class FaissRetrievalConfig(RetrievalConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.retrieval.FaissRetrieval"
    embeddings: EmbeddingsConfig = MISSING


@dataclass
class BM25RetrievalConfig(RetrievalConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.retrieval.BM25Retrieval"


cs = ConfigStore.instance()
cs.store(name="faiss", group="retrieval", node=FaissRetrievalConfig)
cs.store(name="bm25", group="retrieval", node=BM25RetrievalConfig)
