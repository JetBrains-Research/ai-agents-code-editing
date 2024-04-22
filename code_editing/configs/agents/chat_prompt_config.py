from dataclasses import dataclass

from code_editing.configs.utils import CE_CLASSES_ROOT_PKG


@dataclass
class ChatPromptConfig:
    owner_repo_commit: str = "hwchase17/openai-functions-agent"
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.utils.ChatPromptFactory"
