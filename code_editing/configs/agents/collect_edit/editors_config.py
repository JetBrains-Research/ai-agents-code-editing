from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from code_editing.configs.agents.user_prompt_config import UserPromptConfig, default_user_prompt
from code_editing.configs.utils import CE_CLASSES_ROOT_PKG


@dataclass
class EditorConfig:
    _target_: str = MISSING


@dataclass
class SimpleEditorConfig(EditorConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.graph.collect_edit.editors.SimpleEditor"
    edit_prompt: UserPromptConfig = field(default_factory=default_user_prompt("jbr-code-editing/edit"))


cs = ConfigStore.instance()
cs.store(name="simple_editor", group="graph/editor", node=SimpleEditorConfig)
