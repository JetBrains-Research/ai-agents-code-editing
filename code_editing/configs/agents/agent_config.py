from dataclasses import dataclass, field
from typing import Any, Dict

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from code_editing.configs.agents.agent_executor_config import AgentExecutorConfig
from code_editing.configs.agents.chat_prompt_config import ChatPromptConfig
from code_editing.configs.agents.graph_config import GraphConfig
from code_editing.configs.agents.llm_config import ChatLLMConfig
from code_editing.configs.agents.loader_config import LoaderConfig
from code_editing.configs.agents.retrieval_config import RetrievalConfig
from code_editing.configs.agents.tools_config import EditToolConfig, ViewFileToolConfig, CodeSearchToolConfig
from code_editing.configs.backbones_configs import BackboneConfig
from code_editing.configs.data_source_configs import DataSourceConfig
from code_editing.configs.inference_config import InferenceConfig


@dataclass
class RunAgentConfig:
    data_source: DataSourceConfig = MISSING
    tools: Dict[Any, dict] = field(default_factory=dict)
    llm: ChatLLMConfig = MISSING
    chat_prompt: ChatPromptConfig = field(default_factory=ChatPromptConfig)
    graph: GraphConfig = MISSING
    agent_executor: AgentExecutorConfig = field(default_factory=AgentExecutorConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    backbone: BackboneConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="base_agent", node=RunAgentConfig)
# All tool options
cs = ConfigStore.instance()
cs.store(name="edit", group="tools", node=EditToolConfig)
cs.store(name="view_file", group="tools", node=ViewFileToolConfig)
cs.store(name="code_search", group="tools", node=CodeSearchToolConfig)
