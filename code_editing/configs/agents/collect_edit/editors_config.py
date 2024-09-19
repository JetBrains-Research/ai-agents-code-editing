from dataclasses import dataclass, field

from code_editing.configs.agents import GraphConfig
from code_editing.configs.agents.user_prompt_config import UserPromptConfig, default_user_prompt
from code_editing.configs.utils import CE_CLASSES_ROOT_PKG


@dataclass
class EditorConfig(GraphConfig):
    pass


@dataclass
class SimpleEditorConfig(EditorConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.collect_edit.editors.SimpleEditor"
    edit_prompt: UserPromptConfig = field(default_factory=default_user_prompt("jbr-code-editing/edit"))


def setup_editor_config(cs):
    cs.store(name="simple_editor", group="graph/editor", node=SimpleEditorConfig)
