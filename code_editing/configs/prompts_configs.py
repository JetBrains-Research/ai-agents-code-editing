from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from code_editing.configs.utils import CE_CLASSES_ROOT_PKG


@dataclass
class PromptConfig:
    """Base config for instantiating a prompt. Should be extended for each case."""

    _target_: str = MISSING
    max_new_tokens: int = 3000


@dataclass
class ZeroShotPromptConfig(PromptConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.prompts.ZeroShotCEPrompt"


@dataclass
class FewShotPromptConfig(PromptConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.prompts.FewShotCEPrompt"


@dataclass
class FewShotPrompt2Config(PromptConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.prompts.FewShotCEPrompt2"


@dataclass
class CodeFragmentPromptConfig(PromptConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.prompts.CodeFragmentCEPrompt"


@dataclass
class SimpleAgentPromptConfig(PromptConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.agents.prompts.SimpleAgentPrompt"


cs = ConfigStore.instance()
# all available options for the prompt
cs.store(name="zeroshot", group="backbone/prompt", node=ZeroShotPromptConfig)
cs.store(name="fewshot", group="backbone/prompt", node=FewShotPromptConfig)
cs.store(name="fewshot2", group="backbone/prompt", node=FewShotPrompt2Config)
cs.store(name="code_fragment", group="backbone/prompt", node=CodeFragmentPromptConfig)
cs.store(name="simple_agent", group="backbone/prompt", node=SimpleAgentPromptConfig)
