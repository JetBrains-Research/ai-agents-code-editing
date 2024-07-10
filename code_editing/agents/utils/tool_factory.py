from typing import List

from hydra.utils import instantiate
from langchain.tools import BaseTool


class ToolFactory:
    def __init__(self, tool_config_dict: dict):
        self.tool_config_list = tool_config_dict.values()
        self.preview = self._preview()
        self.global_tools_config = {}

    def build(self, *args, **kwargs) -> List[BaseTool]:
        res = []
        for tool_config in self.tool_config_list:
            tool = instantiate(tool_config, *args, **self.global_tools_config, **kwargs)
            res.append(tool)
        return res

    def _preview(self) -> List[BaseTool]:
        res = []
        for tool_config in self.tool_config_list:
            res.append(instantiate(tool_config, dry_run=True))
        return res

    @property
    def short_names(self) -> List[str]:
        """Short names detailing the tools in the factory. Used for logging and wandb run name."""
        return [tool.short_name for tool in self.preview if tool.short_name is not None]

    @property
    def short_name(self) -> str:
        """Short name detailing all the tools in the factory. Used for logging and wandb run name."""
        names = self.short_names
        return "_".join([name for name in names if name is not None])
