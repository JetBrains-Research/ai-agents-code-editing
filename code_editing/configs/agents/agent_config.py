from dataclasses import dataclass, field
from typing import Any, Dict

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from code_editing.configs.agents.agent_executor_config import AgentExecutorConfig
from code_editing.configs.agents.chat_prompt_config import ChatPromptConfig
from code_editing.configs.agents.context_providers.context_config import ContextConfig
from code_editing.configs.agents.graph_config import GraphConfig
from code_editing.configs.agents.llm_config import ChatLLMConfig
from code_editing.configs.agents.tools_config import ToolConfig
from code_editing.configs.backbones_configs import BackboneConfig
from code_editing.configs.data_source_configs import DataSourceConfig
from code_editing.configs.inference_config import InferenceConfig


@dataclass
class RunAgentConfig:
    agent_executor: AgentExecutorConfig = field(default_factory=AgentExecutorConfig)
    backbone: BackboneConfig = MISSING
    chat_prompt: ChatPromptConfig = field(default_factory=ChatPromptConfig)
    context: Dict[Any, ContextConfig] = field(default_factory=dict)
    data_source: DataSourceConfig = MISSING
    graph: GraphConfig = MISSING
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    llm: ChatLLMConfig = MISSING
    tools: Dict[Any, ToolConfig] = field(default_factory=dict)


cs = ConfigStore.instance()
cs.store(name="base_agent", node=RunAgentConfig)
