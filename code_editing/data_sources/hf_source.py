import logging
import os
from abc import abstractmethod
from typing import Optional, Dict, List

from datasets import load_dataset
from tqdm import tqdm

from code_editing.backbones.base_backbone import CEInput
from code_editing.data_sources.base_source import CEDataSource
from code_editing.data_sources.extract_code_base import CodeBaseExtractor
from code_editing.data_sources.git_data import SimpleGitCEData
from code_editing.utils.git_utils import clone_repo, get_diff, get_changed_files_patch


class HuggingFaceCEDataSource(CEDataSource):
    """
    Data source for HuggingFace datasets.
    It doesn't implement any data loading logic, but provides a common interface for all HuggingFace datasets.
    """

    def __init__(
        self,
        hub_name: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._dataset = load_dataset(hub_name, config, split=split, cache_dir=cache_dir)
        self.name = hub_name
        self._hub_name = hub_name

    def __iter__(self):
        for i, row in enumerate(self._dataset):
            try:
                yield self.row_to_input(row), row
            except Exception as e:
                logging.warning(f"Error in getting data for row #{i} in {self.name}", exc_info=e)

    def __getitem__(self, idx):
        try:
            row = self._dataset[idx]
            return self.row_to_input(row), row
        except Exception as e:
            logging.warning(f"Error in getting data for row #{idx} in {self.name}", exc_info=e)

    def __len__(self):
        return len(self._dataset)

    @abstractmethod
    def row_to_input(self, row: Dict) -> CEInput:
        """Method that converts a row from the dataset to a CEInput object"""
        pass

    @abstractmethod
    def row_to_diff(self, row: Dict) -> str:
        """Method that converts a row from the dataset to a ground truth diff"""
        pass


class HuggingFaceSimpleGitCEDataSource(HuggingFaceCEDataSource):
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
        **kwargs,
    ):
        super().__init__(hub_name, config, split, cache_dir, **kwargs)
        self._data_path = base_data_path
        self._extractor = extractor
        self._init_data_path()

    def _init_data_path(self):
        # Initialize the git repositories
        os.makedirs(self._data_path, exist_ok=True)
        pbar = tqdm(self._dataset, desc=f"Cloning repositories for {self.name}")
        ok_count = 0
        all_count = len(self._dataset)
        for i, row in enumerate(pbar):
            repo = self._row_to_repo(row)
            pbar.set_postfix_str(f"{repo}")
            clone_repo(repo, self._data_path)
            ok_count += 1
        logging.info(f"Successfully cloned {ok_count}/{all_count} repositories for {self.name}")

    @abstractmethod
    def row_to_data(self, row: Dict) -> SimpleGitCEData:
        """Method that converts a row from the dataset to a SimpleGitCEData object"""
        pass

    def _row_to_repo(self, row: Dict) -> str:
        """Faster method to get the repo name from a row. By default, it uses the data from row_to_data.
        Can be overridden to speed up the init process."""
        return self.row_to_data(row).repo

    def row_to_diff(self, row: Dict) -> str:
        data = self.row_to_data(row)
        return data.diff_true

    def row_to_input(self, row: Dict) -> CEInput:
        data = self.row_to_data(row)
        code_base = self._extractor(data, self._data_path)
        return {"instruction": data.message, "code_base": code_base}

    def row_to_files(self, row: Dict) -> List[str]:
        data = self.row_to_data(row)
        return get_changed_files_patch(data.repo, data.diff_true, self._data_path, data.base_hash)

    def __len__(self):
        return len(self._dataset)

    @property
    def data_path(self):
        return self._data_path

    def full_data(self):
        return self._dataset

    def _get_diff(self, repo, hash, base_hash):
        return get_diff(repo, hash, self._data_path, base_hash)
