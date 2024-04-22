from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ChatLLMConfig:
    _target_: str = MISSING


@dataclass
class OpenAILLMConfigChat(ChatLLMConfig):
    _target_: str = "langchain_openai.ChatOpenAI"
    model_name: str = "gpt-3.5-turbo"


@dataclass
class HuggingFaceLLMConfig:
    _target_: str = "langchain_community.llms.HuggingFaceHub"
    repo_id: str = MISSING
    task: str = "text-generation"


@dataclass
class HuggingFaceChatLLMConfig(ChatLLMConfig):
    _target_: str = "langchain_community.chat_models.ChatHuggingFace"
    llm: HuggingFaceLLMConfig = field(default_factory=HuggingFaceLLMConfig)


cs = ConfigStore.instance()
cs.store(name="openai", group="llm", node=OpenAILLMConfigChat)
cs.store(name="base_hf", group="llm", node=HuggingFaceChatLLMConfig)
cs.store(name="base_hf", group="llm/llm", node=HuggingFaceLLMConfig)
