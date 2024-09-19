from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from code_editing.configs.agents import GraphConfig
from code_editing.configs.agents.collect_edit.context_collectors_config import ContextCollectorsConfig
from code_editing.configs.agents.collect_edit.editors_config import EditorConfig
from code_editing.configs.agents.user_prompt_config import UserPromptConfig, default_user_prompt
from code_editing.configs.utils import CE_CLASSES_ROOT_PKG


@dataclass
class AgentOnlyConfig(GraphConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.end_to_end.AgentOnly"
    agent_prompt: UserPromptConfig = field(default_factory=default_user_prompt("jbr-code-editing/agent"))


@dataclass
class SelfReflectionConfig(GraphConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.end_to_end.SelfReflection"
    agent_prompt: UserPromptConfig = field(default_factory=default_user_prompt("jbr-code-editing/agent"))
    agent_review_prompt: UserPromptConfig = field(
        default_factory=default_user_prompt("jbr-code-editing/agent-reviewed")
    )
    review_prompt: UserPromptConfig = field(default_factory=default_user_prompt("jbr-code-editing/review"))


@dataclass
class CollectEditConfig(GraphConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.collect_edit.CollectEdit"
    context_collector: ContextCollectorsConfig = MISSING
    editor: EditorConfig = MISSING
    only_collect: bool = False


cs = ConfigStore.instance()
cs.store(name="agent_only", group="graph", node=AgentOnlyConfig)
cs.store(name="self_reflection", group="graph", node=SelfReflectionConfig)
cs.store(name="collect_edit", group="graph", node=CollectEditConfig)
