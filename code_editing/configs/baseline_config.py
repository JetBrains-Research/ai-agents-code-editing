from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from code_editing.configs.backbones_configs import BackboneConfig
from code_editing.configs.data_source_configs import DataSourceConfig
from code_editing.configs.extractor_config import ExtractorConfig
from code_editing.configs.inference_config import InferenceConfig
from code_editing.configs.preprocessor_config import PreprocessorConfig, TruncationPreprocessorConfig


@dataclass
class RunBaselineConfig:
    backbone: BackboneConfig = MISSING
    preprocessor: PreprocessorConfig = MISSING
    data_source: DataSourceConfig = MISSING
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


cs = ConfigStore.instance()
cs.store(name="base_baseline", node=RunBaselineConfig)
# all available options for the preprocessor
cs.store(name="truncate", group="preprocessor", node=TruncationPreprocessorConfig)
