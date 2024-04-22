from dataclasses import dataclass

from hydra.core.config_store import ConfigStore


@dataclass
class WandbConfig:
    project: str = "lca-code-editing"
    enable: bool = False


cs = ConfigStore.instance()
cs.store(name="wandb", node=WandbConfig)
