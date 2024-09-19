from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING
from transformers import BitsAndBytesConfig

from code_editing.configs.prompts_configs import PromptConfig, setup_prompt_config
from code_editing.configs.utils import CE_CLASSES_ROOT_PKG


@dataclass
class BackboneConfig:
    """Base config for instantiating a backbone. Should be extended for each case."""

    _target_: str = MISSING
    prompt: Optional[PromptConfig] = None
    model_name: str = MISSING


@dataclass
class HFModelConfig:
    """Config for initializing a HuggingFace model. Includes some options; the rest can be added via Hydra's override
    (e.g., ++backbone.model_kwargs.cache_dir=some_dir).

    All kwargs will be passed to transformers.PreTrainedModel.from_pretrained. See docs here:
    https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
    """

    torch_dtype: str = "auto"
    device_map: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    attn_implementation: Optional[str] = None
    quantization_config: Optional[BitsAndBytesConfig] = None


@dataclass
class HFGenerationConfig:
    """Config for generation via HuggingFace models. Includes some options; the rest can be added via Hydra's override
    (e.g., ++generation.forced_bos_token_id=0).

    All kwargs will be passed to transformers.GenerationConfig. See docs here:
    https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig"""

    do_sample: bool = True
    temperature: float = 0.8
    max_length: Optional[int] = None
    eos_token_id: Optional[int] = None  # If None, tokenizer eos_token_id is used


@dataclass
class HFBackboneConfig(BackboneConfig):
    """Config for instantiating a HuggingFace backbone.

    Attributes:
        model_name: Name of the model on HF Hub or local path to checkpoint.
        prompt: Name for one of the supported prompt configurations (optional, if not given, raw diff will be passed).
        is_encoder_decoder: True for seq2seq models, False for decoder-only models.
        model_kwargs: Config for model initialization.
        generation: Config for generation.
        device: Device to put model & data on
          (docs here: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)
        seed: Seed for reproducibility; if None, will be chosen randomly.
        use_bettertransformer: Set to True to enable BetterTransformer
          (details here: https://huggingface.co/docs/transformers/perf_infer_gpu_one#bettertransformer)
    """

    _target_: str = f"{CE_CLASSES_ROOT_PKG}.utils.backbones.HuggingFaceBackbone"
    is_encoder_decoder: bool = MISSING
    model_kwargs: HFModelConfig = field(default_factory=HFModelConfig)
    generation: HFGenerationConfig = field(default_factory=HFGenerationConfig)
    device: str = "cuda"
    seed: Optional[int] = 42
    use_bettertransformer: bool = False


@dataclass
class DummyBackboneConfig(BackboneConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.utils.backbones.DummyBackbone"
    model_name: str = "gpt-3.5-turbo"  # For token counting, not crucial
    file_sample_p: float = 0.5
    line_sample_p: float = 0.2


@dataclass
class OpenAIBackboneConfig(BackboneConfig):
    _target_: str = f"{CE_CLASSES_ROOT_PKG}.utils.backbones.OpenAIBackbone"


def setup_backbones_config(cs):
    setup_prompt_config(cs)

    cs.store(name="base_hf", group="backbone", node=HFBackboneConfig)
    cs.store(name="dummy", group="backbone", node=DummyBackboneConfig)
    cs.store(name="base_openai", group="backbone", node=OpenAIBackboneConfig)
