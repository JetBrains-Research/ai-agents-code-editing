from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from code_editing.configs.utils import CE_CLASSES_ROOT_PKG


@dataclass
class ToolConfig:
    _target_: str = MISSING


@dataclass
class EditToolConfig(ToolConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.tools.EditTool"


@dataclass
class ViewFileToolConfig(ToolConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.tools.ViewFileTool"


@dataclass
class CodeSearchToolConfig(ToolConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.tools.CodeSearchTool"
    show_outputs: bool = True
    calls_limit: Optional[int] = None
    do_add_to_viewed: bool = False


# ACR boilerplate
@dataclass
class ACRSearchClassConfig(ToolConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.tools.ACRSearchClass"


@dataclass
class ACRSearchMethodInFileConfig(ToolConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.tools.ACRSearchMethodInFile"


@dataclass
class ACRSearchMethodInClassConfig(ToolConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.tools.ACRSearchMethodInClass"


@dataclass
class ACRSearchMethodConfig(ToolConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.tools.ACRSearchMethod"


@dataclass
class ACRSearchCodeConfig(ToolConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.tools.ACRSearchCode"


@dataclass
class ACRSearchCodeInFileConfig(ToolConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.tools.ACRSearchCodeInFile"


@dataclass
class ACRShowDefinitionConfig(ToolConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.tools.ACRShowDefinition"


@dataclass
class AiderRepoMapConfig(ToolConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.tools.RepoMapTool"


def setup_tools_config(cs):
    # All tool options
    cs.store(name="edit", group="tools", node=EditToolConfig)
    cs.store(name="view_file", group="tools", node=ViewFileToolConfig)
    cs.store(name="code_search", group="tools", node=CodeSearchToolConfig)

    # All ACR tool options
    cs.store(name="acr_search_class", group="tools", node=ACRSearchClassConfig)
    cs.store(name="acr_search_method_in_file", group="tools", node=ACRSearchMethodInFileConfig)
    cs.store(name="acr_search_method_in_class", group="tools", node=ACRSearchMethodInClassConfig)
    cs.store(name="acr_search_method", group="tools", node=ACRSearchMethodConfig)
    cs.store(name="acr_search_code", group="tools", node=ACRSearchCodeConfig)
    cs.store(name="acr_search_code_in_file", group="tools", node=ACRSearchCodeInFileConfig)
    cs.store(name="acr_show_definition", group="tools", node=ACRShowDefinitionConfig)

    cs.store(name="repo_map", group="tools", node=AiderRepoMapConfig)
