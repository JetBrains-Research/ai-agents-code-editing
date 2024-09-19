from dataclasses import dataclass, field
from typing import Optional

from code_editing.configs.wandb_config import WandbConfig, setup_wandb_config


@dataclass
class InferenceConfig:
    checkpoint_iters: Optional[int] = None
    start_from: int = 0
    end_at: Optional[int] = None
    num_workers: int = 5
    output_path: Optional[str] = None
    wandb: WandbConfig = field(default_factory=WandbConfig)
    num_tries: int = 5
    skip_empty_diffs: bool = True
    run_name: Optional[str] = None
    run_suffix: str = ""
    run_prefix: str = ""


def setup_inference_config(cs):
    setup_wandb_config(cs)
