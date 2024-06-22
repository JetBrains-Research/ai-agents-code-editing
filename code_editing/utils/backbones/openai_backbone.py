import logging

from openai import OpenAI

from code_editing.code_editor import CEBackbone, CEInput, CEOutput
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
        preprocessed_inputs = self._prompt.chat(req)

        def openai_request(inp):
            resp = self.api.chat.completions.create(
                messages=inp,
                model=self._model_name,
            )
            return str(resp.choices[0].message.content or "")

        return self._prompt.postprocess(req, {"prediction": openai_request(preprocessed_inputs)})
