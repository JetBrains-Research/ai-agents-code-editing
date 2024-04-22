from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from code_editing.configs.utils import CE_CLASSES_ROOT_PKG


@dataclass
class ToolConfig:
    _target_: str = MISSING


@dataclass
class EditToolConfig:
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.tools.EditTool"


@dataclass
class ViewFileToolConfig:
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.tools.ViewFileTool"


@dataclass
class CodeSearchToolConfig:
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.tools.CodeSearchTool"
    show_outputs: bool = True
    calls_limit: Optional[int] = None
    do_add_to_viewed: bool = True
