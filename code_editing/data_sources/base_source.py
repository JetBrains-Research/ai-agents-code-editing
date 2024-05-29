from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Sequence, Sized

from code_editing.backbones.base_backbone import CEInput
from code_editing.data_sources.extract_code_base import CodeBaseExtractor
from code_editing.data_sources.git_data import SimpleGitCEData


class CEDataSource(ABC, Sequence[SimpleGitCEData]):
    """Code editing data source. Iterates over git data."""

    name = "base"

    def __init__(self, extractor: CodeBaseExtractor, data_path: str):
        self._extractor = extractor
        self.data_path = data_path

    def data_to_input(self, data: SimpleGitCEData) -> CEInput:
        code_base = self._extractor(data, self.data_path)
        return {"instruction": data.message, "code_base": code_base}

    @abstractmethod
    def get_lock(self, item):
        """Lock context for working with the data point. Should be acquired before using __getitem__"""
        return nullcontext()

    @abstractmethod
    def all_locks(self):
        """List of locks for all data points. Should be acquired before using __getitem__"""
        return []
