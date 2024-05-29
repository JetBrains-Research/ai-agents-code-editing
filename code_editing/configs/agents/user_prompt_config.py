from dataclasses import dataclass
from typing import Callable, Optional

from code_editing.configs.utils import CE_CLASSES_ROOT_PKG


@dataclass
class UserPromptConfig:
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.utils.PromptWrapper"
    template: Optional[str] = None
    template_file: Optional[str] = None
    owner_repo_commit: Optional[str] = None
    overrides: Optional[dict] = None


def default_user_prompt(owner_repo_commit: Optional[str] = None) -> Callable[[], UserPromptConfig]:
    return lambda: UserPromptConfig(owner_repo_commit=owner_repo_commit)
