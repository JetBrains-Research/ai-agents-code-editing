from dataclasses import dataclass, field

from omegaconf import MISSING

from code_editing.configs.backbones_configs import BackboneConfig, setup_backbones_config
from code_editing.configs.data_source_configs import DataSourceConfig, setup_data_source_config
from code_editing.configs.extractor_config import ExtractorConfig, setup_extractor_config
from code_editing.configs.inference_config import InferenceConfig, setup_inference_config
from code_editing.configs.preprocessor_config import PreprocessorConfig, TruncationPreprocessorConfig


@dataclass
class RunBaselineConfig:
    backbone: BackboneConfig = MISSING
    preprocessor: PreprocessorConfig = MISSING
    data_source: DataSourceConfig = MISSING
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def setup_baseline_config(cs):
    setup_backbones_config(cs)
    setup_data_source_config(cs)
    setup_extractor_config(cs)
    setup_inference_config(cs)

    cs.store(name="base_baseline", node=RunBaselineConfig)
    cs.store(name="truncate", group="preprocessor", node=TruncationPreprocessorConfig)
