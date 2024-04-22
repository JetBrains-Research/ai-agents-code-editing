from dataclasses import dataclass
from typing import Optional

from code_editing.configs.utils import CE_CLASSES_ROOT_PKG


@dataclass
class UserPromptConfig:
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.utils.PromptWrapper"
    template: Optional[str] = None
    template_file: Optional[str] = None
    owner_repo_commit: Optional[str] = None
    overrides: Optional[dict] = None
