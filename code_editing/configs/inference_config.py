from dataclasses import dataclass, field
from typing import Optional

from code_editing.configs.wandb_config import WandbConfig


@dataclass
class InferenceConfig:
    checkpoint_iters: Optional[int] = None
    start_from: int = 0
    end_at: Optional[int] = None
    num_workers: int = 5
    output_path: Optional[str] = None
    wandb: WandbConfig = field(default_factory=WandbConfig)
    num_tries: int = 5
