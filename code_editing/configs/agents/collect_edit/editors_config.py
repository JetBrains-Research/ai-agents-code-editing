from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from code_editing.configs.agents.user_prompt_config import UserPromptConfig
from code_editing.configs.utils import CE_CLASSES_ROOT_PKG


@dataclass
class EditorConfig:
    _target_: str = MISSING


@dataclass
class SimpleEditorConfig(EditorConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.graph.collect_edit.editors.SimpleEditor"
    edit_prompt: UserPromptConfig = field(default_factory=UserPromptConfig)


cs = ConfigStore.instance()
cs.store(name="simple_editor", group="graph/editor", node=SimpleEditorConfig)
