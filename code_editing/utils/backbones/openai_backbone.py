import logging
from typing import Optional

from openai import OpenAI
from wandb.sdk.data_types.trace_tree import Trace

from code_editing.code_editor import CEBackbone, CEInput, CEOutput
from code_editing.utils import wandb_utils
from code_editing.utils.prompts.base_prompt import CEPrompt


class OpenAIBackbone(CEBackbone):
    """
    This is a backbone that uses OpenAI API to generate a diff.
    """

    def __init__(self, model_name: str, prompt: CEPrompt, **kwargs):
        super().__init__()
        self._model_name = model_name
        self._prompt = prompt
        self.api = OpenAI(api_key=kwargs.get("api_key", None))
        self.name = f"openai/{model_name}"
        # Disable OpenAI API logging
        logging.getLogger("httpx").setLevel(logging.WARNING)

    def generate_diff(self, req: CEInput, **kwargs) -> CEOutput:
        # Initialize the root span for W&B
        parent_span: Optional[Trace] = kwargs.get("parent_span", None)

        preprocessed_inputs = wandb_utils.log_prompt_trace(
            parent_span,
            metadata={
                "prompt_name": self._prompt.name,
            },
        )(
            self._prompt.chat
        )(req)

        @wandb_utils.log_llm_trace(parent_span=parent_span, model_name=self._model_name)
        def openai_request(inp):
            resp = self.api.chat.completions.create(
                messages=inp,
                model=self._model_name,
            )
            return str(resp.choices[0].message.content or "")

        return self._prompt.postprocess(req, {"prediction": openai_request(preprocessed_inputs)})
