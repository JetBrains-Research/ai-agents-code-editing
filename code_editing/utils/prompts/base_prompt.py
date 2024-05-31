import logging
from abc import ABC, abstractmethod
from typing import List, Optional, TypedDict

from jinja2.exceptions import TemplateError
from transformers import PreTrainedTokenizerFast

from code_editing.code_editor import CEInput, CEOutput


class ChatMessage(TypedDict):
    role: str
    content: str


class PromptStatistics(TypedDict):
    original_length: int
    processed_length: int


DEFAULT_HF_CHAT_TEMPLATE = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message["
    "'content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ "
    "'<|im_start|>assistant\n' }}{% endif %}"
)


def has_chat_template(tokenizer) -> bool:
    return tokenizer.chat_template is not None or tokenizer.default_chat_template != DEFAULT_HF_CHAT_TEMPLATE


class CEPrompt(ABC):
    name = "base"

    def __init__(self, max_new_tokens: int):
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def chat(self, req: CEInput, has_system_prompt: bool = True) -> List[ChatMessage]:
        pass

    def _to_raw(self, req: CEInput) -> str:
        """Fallback method to convert a CEInput to a raw string. Last resort."""
        res = ""
        for m in self.chat(req):
            res += f"### {m['role'].capitalize()} ###\n"
            res += m["content"]
            res += "\n\n"
        return res

    def hf(
        self,
        req: CEInput,
        prompt_format: Optional[str],
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
    ) -> str:
        if not prompt_format and tokenizer and has_chat_template(tokenizer):
            logging.info("Using chat template from HF tokenizer.")
            chat_messages = self.chat(req)
            try:
                return tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
            except TemplateError:
                chat_messages_no_sys = self.chat(req, has_system_prompt=False)
                try:
                    logging.warning("Couldn't use chat template with system prompt. Trying without the system prompt.")
                    return tokenizer.apply_chat_template(
                        chat_messages_no_sys, tokenize=False, add_generation_prompt=True
                    )
                except TemplateError:
                    logging.error("Couldn't apply chat template. Concatenating system message with user message.")
                    chat_messages = [{"role": "user", "content": self._to_raw(req)}]
                    return tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)

        if not prompt_format:
            logging.warning("Prompt format not passed, using raw diff as model input.")
            return self._to_raw(req)

        if prompt_format == "starchat":  # https://huggingface.co/HuggingFaceH4/starchat-beta#intended-uses--limitations
            chat_messages = self.chat(req)
            assert (
                chat_messages[0]["role"] == "system" and chat_messages[1]["role"] == "user"
            ), "By default, a single system message is expected."
            res = [f"<|{message['role']}|>\n{message['content']}<|end|>" for message in chat_messages]
            res.append("<|assistant|>")
            return "\n".join(res)

        if prompt_format == "llama":  # https://huggingface.co/blog/codellama#conversational-instructions
            chat_messages = self.chat(req)
            assert (
                chat_messages[0]["role"] == "system" and chat_messages[1]["role"] == "user"
            ), "By default, a single system message followed by a user message are expected."

            res = f"<s>[INST] <<SYS>>\n{chat_messages[0]['content']}\n<</SYS>>\n\n{chat_messages[1]['content']}[/INST] "
            for i in range(2, len(chat_messages), 2):
                assistant_answer = chat_messages[i]["content"]
                user_req = chat_messages[i + 1]["content"]
                res += f"{assistant_answer} </s><s>[INST] {user_req} [/INST] "
            return res

        raise NotImplementedError("Unknown prompt format.")

    def postprocess(self, req: CEInput, resp: CEOutput) -> CEOutput:
        """Postprocess the response. This method can be overridden by subclasses."""
        return resp
