from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from code_editing.configs.agents.context_providers.loader_config import LoaderConfig
from code_editing.configs.agents.embeddings_config import EmbeddingsConfig
from code_editing.configs.utils import CE_CLASSES_ROOT_PKG


@dataclass
class ContextConfig:
    _target_: str = MISSING


@dataclass
class ACRSearchManagerConfig(ContextConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.context_providers.acr_search.SearchManager"
    show_lineno: bool = True


@dataclass
class RetrievalConfig(ContextConfig):
    splitter: Any = MISSING
    loader: LoaderConfig = field(default_factory=LoaderConfig)


@dataclass
class FaissRetrievalConfig(RetrievalConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.context_providers.retrieval.FaissRetrieval"
    embeddings: EmbeddingsConfig = MISSING


@dataclass
class BM25RetrievalConfig(RetrievalConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.context_providers.retrieval.BM25Retrieval"


@dataclass
class AiderRepoMapConfig(ContextConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.context_providers.aider.AiderRepoMap"


def setup_context_config(cs):
    cs.store(name="context", node=ContextConfig)
    cs.store(name="acr_search", group="context", node=ACRSearchManagerConfig)
    cs.store(name="faiss", group="context", node=FaissRetrievalConfig)
    cs.store(name="bm25", group="context", node=BM25RetrievalConfig)
    cs.store(name="aider", group="context", node=AiderRepoMapConfig)
