from dataclasses import dataclass, field
from typing import Any, Dict

from omegaconf import MISSING

from code_editing.configs.data_source_configs import DataSourceConfig, setup_data_source_config
from code_editing.configs.extractor_config import ExtractorConfig, setup_extractor_config
from code_editing.configs.wandb_config import WandbConfig, setup_wandb_config


@dataclass
class RunEvaluationConfig:
    input_path: str = MISSING
    data_source: DataSourceConfig = MISSING
    metrics: Dict[Any, Dict] = field(default_factory=dict)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)


def setup_evaluation_config(cs):
    setup_data_source_config(cs)
    setup_wandb_config(cs)
    setup_extractor_config(cs)

    cs.store(name="base_eval", node=RunEvaluationConfig)
