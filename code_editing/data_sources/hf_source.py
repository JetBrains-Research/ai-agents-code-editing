import logging
import os
from abc import abstractmethod
from typing import Dict, Optional

from datasets import load_dataset
from tqdm import tqdm

from code_editing.data_sources.base_source import CEDataSource
from code_editing.data_sources.extract_code_base import CodeBaseExtractor
from code_editing.data_sources.git_data import SimpleGitCEData
from code_editing.utils.git_utils import clone_repo, get_diff, get_repo_path, lock_repo

logger = logging.getLogger("data_sources")


class HuggingFaceSimpleGitCEDataSource(CEDataSource):
    """
    Data source for HuggingFace datasets that contain commits from git repositories.
    It implements data loading logic for such datasets.
    """

    def __init__(
        self,
        hub_name: str,
        base_data_path: str,
        extractor: CodeBaseExtractor,
        config: Optional[str] = None,
        split: Optional[str] = None,
        cache_dir: Optional[str] = None,
        shuffle_seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(extractor, base_data_path)
        self._dataset = load_dataset(hub_name, config, split=split, cache_dir=cache_dir)

        if shuffle_seed is not None:
            self._dataset = self._dataset.shuffle(shuffle_seed)

        self.name = hub_name
        self.split = split
        self._hub_name = hub_name

        # Initialize the git repositories
        os.makedirs(self.data_path, exist_ok=True)
        pbar = tqdm(self._dataset, desc=f"Cloning repositories for {self.name}")
        all_count = len(self._dataset)
        for row in pbar:
            repo = self._row_to_repo(row)
            pbar.set_postfix_str(f"{repo}")
            clone_repo(repo, self.data_path)
        logger.info(f"Successfully cloned {all_count} repositories for {self.name}")

    def __getitem__(self, item) -> SimpleGitCEData:
        return self.row_to_data(self._dataset[item])

    def __len__(self):
        return len(self._dataset)

    @abstractmethod
    def row_to_data(self, row: Dict) -> SimpleGitCEData:
        """Method that converts a row from the dataset to a SimpleGitCEData object"""
        pass

    @abstractmethod
    def _row_to_repo(self, row: Dict) -> str:
        """Faster method to get the repo name from a row. It must not use git operations."""
        raise NotImplementedError

    def get_lock(self, item):
        repo = self._row_to_repo(self._dataset[item])
        return lock_repo(get_repo_path(self.data_path, repo), self.data_path)

    def all_locks(self):
        repos = set([self._row_to_repo(row) for row in self._dataset])
        return [lock_repo(get_repo_path(self.data_path, repo), self.data_path) for repo in repos]

    def _get_diff_helper(self, repo, hash, base_hash):
        return get_diff(repo, hash, self.data_path, base_hash)
