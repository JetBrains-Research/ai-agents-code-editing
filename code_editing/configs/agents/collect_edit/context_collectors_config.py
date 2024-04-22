from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from code_editing.configs.agents.user_prompt_config import UserPromptConfig
from code_editing.configs.utils import CE_CLASSES_ROOT_PKG


@dataclass
class ContextCollectorsConfig:
    _target_: str = MISSING


@dataclass
class AsIsRetrievalConfig(ContextCollectorsConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.graph.collect_edit.context_collectors.AsIsRetrieval"
    k: Optional[int] = 5
    total_context: Optional[int] = None


@dataclass
class LLMRetrievalConfig(ContextCollectorsConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.graph.collect_edit.context_collectors.LLMRetrieval"
    search_prompt: UserPromptConfig = MISSING
    do_review: bool = True


@dataclass
class LLMCycleRetrievalConfig(LLMRetrievalConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.graph.collect_edit.context_collectors.LLMCycleRetrieval"
    review_prompt: UserPromptConfig = MISSING

@dataclass
class LLMFixedCtxRetrievalConfig(LLMRetrievalConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.graph.collect_edit.context_collectors.LLMFixedCtxRetrieval"
    total_context: int = 10000


cs = ConfigStore.instance()
cs.store(name="as_is_retrieval", group="graph/context_collector", node=AsIsRetrievalConfig)
cs.store(name="llm_retrieval", group="graph/context_collector", node=LLMRetrievalConfig)
cs.store(name="llm_cycle_retrieval", group="graph/context_collector", node=LLMCycleRetrievalConfig)
cs.store(name="llm_fixed_ctx_retrieval", group="graph/context_collector", node=LLMFixedCtxRetrievalConfig)
