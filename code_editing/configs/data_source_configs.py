import os
from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from code_editing.configs.utils import CE_CLASSES_ROOT_PKG


@dataclass
class DataSourceConfig:
    _target_: str = MISSING
    base_data_path: str = os.getenv(
        "CE_DATA_PATH", MISSING
    )  # Directory with dataset repositories. Load using `load_data/load_data_from_hf.py`
    cache_dir: Optional[str] = None
    shuffle_seed: Optional[int] = None


@dataclass
class LCACodeEditingDataSourceConfig(DataSourceConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.data_sources.LCACodeEditingDataSource"


@dataclass
class CIFixPythonDataSourceConfig(DataSourceConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.data_sources.CIFixPythonDataSource"


@dataclass
class LCABugLocalizationDataSourceConfig(DataSourceConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.data_sources.LCABugLocalizationDataSource"


@dataclass
class SWEBenchDataSourceConfig(DataSourceConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.data_sources.SWEBenchDataSource"
    hub_name: str = "princeton-nlp/SWE-bench_Lite"
    split_name: str = "test"


cs = ConfigStore.instance()
# all available options for the data source
cs.store(name="lca_code_editing", group="data_source", node=LCACodeEditingDataSourceConfig)
cs.store(name="ci_fix_python", group="data_source", node=CIFixPythonDataSourceConfig)
cs.store(name="lca_bug_localization", group="data_source", node=LCABugLocalizationDataSourceConfig)
cs.store(name="swe_bench", group="data_source", node=SWEBenchDataSourceConfig)
