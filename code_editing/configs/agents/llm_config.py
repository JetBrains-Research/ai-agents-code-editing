import os
from dataclasses import dataclass

from grazie.api.client.endpoints import GrazieApiGatewayUrls
from grazie.api.client.gateway import AuthType
from grazie.api.client.profiles import LLMProfile, Profile
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ChatLLMConfig:
    _target_: str = MISSING


@dataclass
class OpenAILLMConfigChat(ChatLLMConfig):
    _target_: str = "langchain_openai.ChatOpenAI"
    model_name: str = "gpt-3.5-turbo"


class GrazieLLMConfig(ChatLLMConfig):
    _target_: str = "grazie_langchain_utils.language_models.grazie.ChatGrazie"
    grazie_jwt_token: str = os.environ.get("GRAZIE_JWT_TOKEN", MISSING)
    auth_type: str = AuthType.USER
    client_url: str = GrazieApiGatewayUrls.STAGING
    profile: LLMProfile = Profile.OPENAI_CHAT_GPT


cs = ConfigStore.instance()
cs.store(name="openai", group="llm", node=OpenAILLMConfigChat)
cs.store(name="grazie", group="llm", node=GrazieLLMConfig)


# noinspection PyProtectedMember
def create_agent_method(llm_config: ChatLLMConfig):
    target = llm_config._target_
    if target == "langchain_openai.ChatOpenAI":
        return "langchain.agents.create_openai_tools_agent"
    elif target == "grazie_langchain_utils.language_models.grazie.ChatGrazie":
        return "grazie_langchain_utils.agents.create_grazie_tools_agent"
    else:
        raise ValueError(f"Unknown target: {target}")
