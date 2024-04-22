import os
from abc import ABC
from typing import Any, Optional

from langchain_core.tools import BaseTool


class CEBaseTool(BaseTool, ABC):
    dry_run: bool = False
    data_path: str = None
    repo_path: str = None
    retrieval_helper: Any = None
    handle_tool_error = True
    handle_validation_error = True

    def __init__(
        self, data_path: str = None, repo_path: str = None, retrieval_helper=None, dry_run: bool = False, **kwargs
    ):
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

        if data_path is None:
            raise ValueError("Data path is required")
        if repo_path is None:
            raise ValueError("Repo path is required")

        self.data_path = data_path
        self.repo_path = repo_path
        self.retrieval_helper = retrieval_helper

        if not os.path.exists(self.repo_path) or not os.path.isdir(self.repo_path):
            raise FileNotFoundError(f"Repo path {self.repo_path} does not exist")

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
