import logging
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, set_seed
from wandb.sdk.data_types.trace_tree import Trace

from code_editing.configs.backbones_configs import HFGenerationConfig, HFModelConfig
from code_editing.prompts.base_prompt import CEPrompt

from ..utils import wandb_utils
from .base_backbone import CEBackbone, CEInput, CEOutput


class HuggingFaceBackbone(CEBackbone):
    """
    This is a backbone that uses HuggingFace models to generate a diff.

    It supports both seq2seq and causal models.

    By default, it uses gpu inference optimizations from https://huggingface.co/docs/transformers/perf_infer_gpu_one#combine-optimizations
    """

    MODEL_NAME_TO_PROMPT_FORMAT: Dict[str, str] = {
        "bigcode/octocoder": "octocoder",
        "HuggingFaceH4/starchat-beta": "starchat",
        "HuggingFaceH4/starchat-alpha": "starchat",
        "Salesforce/instructcodet5p-16b": "alpaca",
    }

    def __init__(
        self,
        model_name: str,
        is_encoder_decoder: bool,
        model_kwargs: HFModelConfig,
        generation: HFGenerationConfig,
        device: str,
        prompt: CEPrompt,
        use_bettertransformer: bool,
        seed: int,
    ):
        set_seed(seed)

        self._is_encoder_decoder = is_encoder_decoder
        self._name_or_path = model_name
        self.name = model_name

        if self._is_encoder_decoder:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self._name_or_path, **model_kwargs  # type: ignore[arg-type]
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self._name_or_path, **model_kwargs  # type: ignore[arg-type]
            )
        if use_bettertransformer:
            try:
                self._model = self._model.to_bettertransformer()
            except:
                logging.warning(
                    "Couldn't convert the model to BetterTransformer, proceeding with default implementation."
                )
        self._tokenizer = AutoTokenizer.from_pretrained(self._name_or_path)
        self._tokenizer.use_default_system_prompt = False
        self._model.eval()
        self._device = device

        if generation.max_length is None and prompt.max_new_tokens is None:
            generation.max_length = self._tokenizer.model_max_length
            logging.warning(
                f"Neither `max_length` nor `max_new_tokens` are passed, setting `max_length` to `model_max_length` of "
                f"corresponding tokenizer ({generation.max_length})"
            )
        if generation.eos_token_id is None:
            generation.eos_token_id = self._tokenizer.eos_token_id
        self._generation_config = GenerationConfig(
            **generation,  # type: ignore
            max_new_tokens=prompt.max_new_tokens,
        )
        self._prompt = prompt

    @torch.inference_mode()
    def generate_diff(self, req: CEInput, **kwargs) -> CEOutput:
        if not self._prompt:
            raise ValueError("Prompt is required for HuggingFace models.")

        # Initialize the root span for W&B
        parent_span: Optional[Trace] = kwargs.get("parent_span", None)

        @wandb_utils.log_prompt_trace(
            parent_span,
            metadata={
                "prompt_name": self._prompt.name,
            },
        )
        def get_inp(r):
            return self._prompt.hf(
                r,
                prompt_format=self.MODEL_NAME_TO_PROMPT_FORMAT.get(self._name_or_path, None),
                tokenizer=self._tokenizer,
            )

        preprocessed_inputs = get_inp(req)
        encoding = self._tokenizer(preprocessed_inputs, return_tensors="pt").to(self._device)

        @wandb_utils.log_llm_trace(
            parent_span=parent_span,
            model_name=self._name_or_path,
            metadata={
                "model_config": self._model.config.to_dict(),
                "generation_config": self._generation_config.to_dict(),
            },
        )
        def get_resp(_):
            return self._model.generate(
                **encoding,
                generation_config=self._generation_config,
            )

        predictions = get_resp(preprocessed_inputs)

        # trim context and leave only generated part (not necessary for seq2seq models, bc context is supposed to go
        # to encoder)
        if not self._is_encoder_decoder:
            predictions = predictions[:, encoding.input_ids.shape[1] :]

        decoded_predictions = self._tokenizer.batch_decode(predictions, skip_special_tokens=True)[0]
        return self._prompt.postprocess(req, {"prediction": decoded_predictions})
