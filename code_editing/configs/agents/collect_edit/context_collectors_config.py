from dataclasses import dataclass, field
from typing import Optional

from code_editing.configs.agents import GraphConfig
from code_editing.configs.agents.user_prompt_config import UserPromptConfig, default_user_prompt
from code_editing.configs.utils import CE_CLASSES_ROOT_PKG


@dataclass
class ContextCollectorsConfig(GraphConfig):
    pass


@dataclass
class AsIsRetrievalConfig(ContextCollectorsConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.collect_edit.context_collectors.AsIsRetrieval"
    k: Optional[int] = 5
    total_context: Optional[int] = None


@dataclass
class LLMRetrievalConfig(ContextCollectorsConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.collect_edit.context_collectors.LLMRetrieval"
    search_prompt: UserPromptConfig = field(default_factory=default_user_prompt("jbr-code-editing/search-reviewed"))
    do_review: bool = True


@dataclass
class LLMCycleRetrievalConfig(LLMRetrievalConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.collect_edit.context_collectors.LLMCycleRetrieval"
    review_prompt: UserPromptConfig = field(
        default_factory=default_user_prompt("jbr-code-editing/search-is-sufficient")
    )


@dataclass
class LLMFixedCtxRetrievalConfig(LLMRetrievalConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.collect_edit.context_collectors.LLMFixedCtxRetrieval"
    total_context: int = 10000


@dataclass
class ACRRetrievalConfig(ContextCollectorsConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.collect_edit.context_collectors.ACRRetrieval"
    use_show_definition: bool = False


@dataclass
class MyACRRetrievalConfig(ContextCollectorsConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.collect_edit.context_collectors.MyACRRetrieval"


@dataclass
class AiderRetrievalConfig(ContextCollectorsConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.collect_edit.context_collectors.AiderRetrieval"
    select_prompt: UserPromptConfig = field(default_factory=default_user_prompt("jbr-code-editing/repomap-search"))


def setup_context_collectors_config(cs):
    cs.store(name="as_is_retrieval", group="graph/context_collector", node=AsIsRetrievalConfig)
    cs.store(name="llm_retrieval", group="graph/context_collector", node=LLMRetrievalConfig)
    cs.store(name="llm_cycle_retrieval", group="graph/context_collector", node=LLMCycleRetrievalConfig)
    cs.store(name="llm_fixed_ctx_retrieval", group="graph/context_collector", node=LLMFixedCtxRetrievalConfig)

    cs.store(name="acr", group="graph/context_collector", node=ACRRetrievalConfig)
    cs.store(name="my_acr", group="graph/context_collector", node=MyACRRetrievalConfig)

    cs.store(name="aider", group="graph/context_collector", node=AiderRetrievalConfig)
