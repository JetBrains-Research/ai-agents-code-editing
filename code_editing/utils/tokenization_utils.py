import logging
from typing import Dict, List

import tiktoken
from transformers import AutoTokenizer


class TokenizationUtils:
    """A wrapper for two tokenization-related operations:
    - estimating the number of tokens for a prompt
    - truncating a prompt to first X tokens.
    """

    def __init__(self, profile_name: str):
        model_info = None
        if "gpt" in profile_name:
            model_info = {"model_provider": "openai", "model_name": profile_name}
        if not model_info:
            logging.warning(f"Unknown profile {profile_name}. Will treat it as a model name on HuggingFace Hub.")
            model_info = {"model_provider": "huggingface", "model_name": profile_name}

        self._model_provider = model_info["model_provider"]
        self._model_name = model_info["model_name"]

        if self._model_provider == "openai":
            self._tokenizer = tiktoken.encoding_for_model(self._model_name)
        elif self._model_provider == "huggingface":
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

    def _count_tokens_completion(self, text: str) -> int:
        """Estimates the number of tokens for a given string."""
        if self._model_provider == "openai":
            return len(self._tokenizer.encode(text))

        if self._model_provider == "huggingface":
            return len(self._tokenizer(text).input_ids)

        raise ValueError(f"{self._model_provider} is currently not supported for token estimation.")

    def _count_tokens_chat(self, messages: List[Dict[str, str]]) -> int:
        """Estimates the number of tokens for a given list of messages.

        Note: Currently, for some models (e.g., OpenAI) the returned number might be slightly lower than the actual number of tokens, because the
        special tokens are not considered.
        """
        return sum([self._count_tokens_completion(value) for message in messages for key, value in message.items()])

    def _truncate_completion(self, text: str, max_num_tokens: int, skip_special_tokens=False) -> str:
        """Truncates a given string to first `max_num_tokens` tokens.

        1. Encodes string to a list of tokens via corresponding tokenizer.
        2. Truncates the list of tokens to first `max_num_tokens` tokens.
        3. Decodes list of tokens back to a string.
        """
        if self._model_provider == "openai":
            encoding = self._tokenizer.encode(text)[:max_num_tokens]
            return self._tokenizer.decode(encoding)
        if self._model_provider == "huggingface":
            encoding = self._tokenizer(text).input_ids[:max_num_tokens]
            return self._tokenizer.decode(encoding, skip_special_tokens=skip_special_tokens)

        raise ValueError(f"{self._model_provider} is currently not supported for prompt truncation.")

    def _truncate_chat(self, messages: List[Dict[str, str]], max_num_tokens: int) -> List[Dict[str, str]]:
        """Truncates a given list of messages to first `max_num_tokens` tokens.

        Note: A current version only truncates a last message, which might not be suitable for all use-cases.
        """
        num_tokens_except_last = self._count_tokens_chat(messages[:-1])
        messages[-1]["content"] = self._truncate_completion(
            messages[-1]["content"], max_num_tokens=max_num_tokens - num_tokens_except_last
        )
        return messages

    def truncate_text(self, text: str, max_num_tokens: int) -> str:
        return self._truncate_completion(text, max_num_tokens, skip_special_tokens=True)

    @property
    def model_name(self) -> str:
        return self._model_name
