from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class GraphConfig:
    _target_: str = MISSING
    _partial_: bool = True
