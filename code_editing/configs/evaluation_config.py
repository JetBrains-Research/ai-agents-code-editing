from dataclasses import dataclass, field
from typing import Any, Dict

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from code_editing.configs.data_source_configs import DataSourceConfig
from code_editing.configs.extractor_config import ExtractorConfig
from code_editing.configs.wandb_config import WandbConfig


@dataclass
class RunEvaluationConfig:
    input_path: str = MISSING
    data_source: DataSourceConfig = MISSING
    metrics: Dict[Any, Dict] = field(default_factory=dict)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)


cs = ConfigStore.instance()
cs.store(name="base_eval", node=RunEvaluationConfig)
