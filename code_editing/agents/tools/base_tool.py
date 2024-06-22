import os
from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.tools import BaseTool, ToolException

from code_editing.agents.run import RunOverviewManager, ToolUseStatus


class CEBaseTool(BaseTool, ABC):
    dry_run: bool = False
    data_path: str = None
    repo_path: str = None
    run_overview_manager: RunOverviewManager = None
    handle_tool_error = True
    handle_validation_error = True

    def __init__(self, run_overview_manager: RunOverviewManager = None, dry_run: bool = False, **kwargs):
        """
        Base Tool for Code Editing tools.

        @param data_path: Path to the data directory
        @param repo_path: Path to the repository directory
        @param retrieval_helper: Retrieval Helper (vector database)
        @param dry_run: If True, the tool will only initialize run-specific parameters (and not repo-specific ones) and will not run
        @param kwargs: Additional parameters for the tool
        """
        super().__init__(**kwargs)

        self.dry_run = dry_run
        if self.dry_run:
            return

        if run_overview_manager is None:
            raise ValueError("RunOverviewManager is required")

        self.run_overview_manager = run_overview_manager
        self.repo_path = run_overview_manager.repo_path
        self.data_path = run_overview_manager.data_path

        if not os.path.exists(self.repo_path) or not os.path.isdir(self.repo_path):
            raise FileNotFoundError(f"Repo path {self.repo_path} does not exist")

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        # Track tool usage
        self.run_overview_manager.log_tool_use(self.name, ToolUseStatus.CALL)
        try:
            # Run the tool
            self._run_tool(*args, **kwargs)
            self.run_overview_manager.log_tool_use(self.name, ToolUseStatus.OK)
        except ToolException:
            # Track tool failure
            self.run_overview_manager.log_tool_use(self.name, ToolUseStatus.FAIL)
            raise
        except Exception:
            # Track tool error
            self.run_overview_manager.log_tool_use(self.name, ToolUseStatus.THROWN)
            raise

    @abstractmethod
    def _run_tool(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def _arun(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError("Async run not implemented")

    @property
    def short_name(self) -> Optional[str]:
        """Short name of the tool. Used for logging and tracing"""
        return None

    def get_ctx_provider(self, ctx_provider_name) -> Any:
        return self.run_overview_manager.get_ctx_provider(ctx_provider_name)
