import collections
from dataclasses import asdict, dataclass
from typing import Dict

from code_editing.agents.context_providers.context_provider import ContextProvider
from code_editing.utils import wandb_utils


@dataclass
class ToolInfo:
    calls: int = 0
    errors: int = 0
    failures: int = 0


class RunOverviewManager:
    def __init__(
        self,
        repo_path: str,
        data_path: str,
        context_providers: Dict[str, ContextProvider],
    ):
        self.repo_path = repo_path
        self.data_path = data_path
        self.context_providers = context_providers
        self.metadata = {"repo_path": repo_path, "data_path": data_path, "context_providers": context_providers.keys()}
        self.tools_info = collections.defaultdict(ToolInfo)
        self.start_ms = wandb_utils.get_current_ms()

    def add_tool_use(self, tool_name):
        self.tools_info[tool_name].calls += 1

    def add_tool_error(self, tool_name):
        self.tools_info[tool_name].errors += 1

    def add_tool_failure(self, tool_name):
        self.tools_info[tool_name].failures += 1

    def get_run_summary(self):
        end_ms = wandb_utils.get_current_ms()
        return {
            "tools": {k: asdict(v) for k, v in self.tools_info.items()},
            "start_ms": self.start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - self.start_ms,
        }

    def get_ctx_provider(self, ctx_provider_name) -> ContextProvider:
        res = self.context_providers.get(ctx_provider_name, None)
        if res is None:
            raise ValueError(f"Context provider {ctx_provider_name} not found")
        return res
