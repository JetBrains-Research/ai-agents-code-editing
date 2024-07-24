import collections
import os
from enum import Enum
from typing import Dict, TypedDict, Union, Type

from hydra.core.hydra_config import HydraConfig

from code_editing.agents.context_providers.context_provider import ContextProvider
from code_editing.utils import wandb_utils


class ToolInfo(TypedDict):
    calls: int
    success: int
    failures: int
    errors: int


# enum class: calls, errors, failures
class ToolUseStatus(Enum):
    CALL = "calls"
    OK = "success"
    FAIL = "failures"
    THROWN = "errors"


class RunOverviewManager:
    def __init__(
        self,
        repo_path: str,
        data_path: str,
        context_providers: Dict[str, ContextProvider],
        instance_id: str = "???",
    ):
        self.repo_path = repo_path
        self.data_path = data_path
        self.context_providers = context_providers
        self.metadata = {"repo_path": repo_path, "data_path": data_path, "context_providers": context_providers.keys()}
        self.tools_info = collections.defaultdict(dict)
        self.start_ms = wandb_utils.get_current_ms()
        self.instance_id = instance_id

    def log_tool_use(self, tool_name, status: ToolUseStatus):
        status = status.value
        self.tools_info.setdefault(tool_name, {}).setdefault(status, 0)
        self.tools_info[tool_name][status] += 1

    def get_run_summary(self):
        end_ms = wandb_utils.get_current_ms()
        return {
            "tools": self.tools_info,
            "duration_sec": (end_ms - self.start_ms) / 1000,
        }

    def get_ctx_provider(self, ctx_provider_name: Union[str, Type[ContextProvider]]) -> ContextProvider:
        if isinstance(ctx_provider_name, str):
            res = self.context_providers.get(ctx_provider_name, None)
            if res is None:
                raise ValueError(f"Context provider {ctx_provider_name} not found")
            return res
        else:
            # filter values by type
            res = list(filter(lambda x: isinstance(x, ctx_provider_name), self.context_providers.values()))
            if len(res) == 0:
                raise ValueError(f"Context provider {ctx_provider_name} not found")
            if len(res) > 1:
                raise ValueError(f"Multiple context providers of type {ctx_provider_name} found")
            return res[0]

    def get_log_path(self) -> str:
        base_path = HydraConfig.get().runtime.output_dir
        log_path = os.path.join(base_path, "logs", f"run_{self.instance_id.replace('/', '_')}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        return log_path
