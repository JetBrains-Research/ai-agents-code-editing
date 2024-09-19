from typing import Any

from pydantic import BaseModel

from code_editing.agents.context_providers.aider import AiderRepoMap
from code_editing.agents.tools.base_tool import CEBaseTool


class RepoMapTool(CEBaseTool):
    class RepoMapToolInput(BaseModel):
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args_schema = self.RepoMapToolInput
        if self.dry_run:
            return
        self.aider = self.get_ctx_provider(AiderRepoMap)

    name = "repo-map"
    description = "Use this tool to show the structure of the repository: files, classes, methods, etc."
    args_schema = RepoMapToolInput

    def _run_tool(self, *args, **kwargs) -> str:
        return self.aider.get_repo_map()

    aider: AiderRepoMap = None
