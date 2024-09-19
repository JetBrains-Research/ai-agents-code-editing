from dataclasses import dataclass, field
from typing import Any, Dict

from omegaconf import MISSING

from code_editing.configs.agents.context_providers.context_config import ContextConfig, setup_context_config
from code_editing.configs.agents.graph_config import GraphConfig, setup_graph_config
from code_editing.configs.agents.llm_config import ChatLLMConfig, setup_llm_config
from code_editing.configs.agents.tools_config import ToolConfig, setup_tools_config
from code_editing.configs.data_source_configs import DataSourceConfig, setup_data_source_config
from code_editing.configs.inference_config import InferenceConfig


@dataclass
class RunAgentConfig:
    context: Dict[Any, ContextConfig] = field(default_factory=dict)
    data_source: DataSourceConfig = MISSING
    graph: GraphConfig = MISSING
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    llm: ChatLLMConfig = MISSING
    tools: Dict[Any, ToolConfig] = field(default_factory=dict)


def setup_agent_config(cs):
    setup_context_config(cs)
    setup_data_source_config(cs)
    setup_graph_config(cs)
    setup_llm_config(cs)
    setup_tools_config(cs)

    cs.store(name="base_agent", node=RunAgentConfig)
