from dataclasses import dataclass

from code_editing.configs.utils import CE_CLASSES_ROOT_PKG


@dataclass
class PreprocessorConfig:
    _target_: str


@dataclass
class TruncationPreprocessorConfig(PreprocessorConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.preprocessors.TruncationCEPreprocessor"
    max_length: int = 6000
