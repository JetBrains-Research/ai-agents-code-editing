from dataclasses import dataclass


@dataclass
class WandbConfig:
    project: str = "lca-code-editing"
    enable: bool = False


def setup_wandb_config(cs):
    cs.store(name="wandb", node=WandbConfig)
