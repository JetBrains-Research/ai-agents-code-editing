from dataclasses import dataclass

from omegaconf import MISSING

from code_editing.configs.utils import CE_CLASSES_ROOT_PKG


@dataclass
class ExtractorConfig:
    _target_: str = MISSING


@dataclass
class FullFileExtractorConfig(ExtractorConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.data_sources.FullFileExtractor"


@dataclass
class CodeFragmentExtractorConfig(ExtractorConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.data_sources.CodeFragmentExtractor"


def setup_extractor_config(cs):
    cs.store(name="full_file", group="extractor", node=FullFileExtractorConfig)
    cs.store(name="code_fragment", group="extractor", node=CodeFragmentExtractorConfig)
