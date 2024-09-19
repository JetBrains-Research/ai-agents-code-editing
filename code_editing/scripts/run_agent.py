from typing import Any, Callable

import dotenv
import hydra
import omegaconf
from hydra.utils import instantiate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableConfig
from tqdm.contrib.logging import logging_redirect_tqdm

from code_editing.agents.agent_graph import AgentGraphPartial

dotenv.load_dotenv()

from code_editing.agents.agent_codeeditor import AgentCodeEditor
from code_editing.agents.utils.checkout_extractor import CheckoutExtractor
from code_editing.agents.utils.tool_factory import ToolFactory
from code_editing.configs.agents.agent_config import RunAgentConfig
from code_editing.data_sources.base_source import CEDataSource
from code_editing.data_sources.hf_source import HuggingFaceSimpleGitCEDataSource
from code_editing.scripts.common import finish_wandb, get_cool_name, inference_loop, init_output_path, init_wandb


@hydra.main(version_base=None, config_path="conf", config_name="agent")
def main(cfg: RunAgentConfig):
    # Initialize extractor and data source
    extractor = CheckoutExtractor()
    data_source: CEDataSource = instantiate(cfg.data_source, extractor=extractor)
    if not isinstance(data_source, HuggingFaceSimpleGitCEDataSource):
        raise ValueError("This script only supports HuggingFaceSimpleGitCEDataSource")
    data_path = data_source.data_path

    # Instantiate agent params
    tool_factory = ToolFactory(cfg.tools)
    tool_factory.global_tools_config = {
        "handle_tool_error": True,
    }
    llm: BaseLanguageModel = instantiate(cfg.llm)

    # Set up the graph for the agent interactions
    agent_graph_partial: AgentGraphPartial = instantiate(cfg.graph, llm=llm, _partial_=True)

    # Set up the tracing tags and metadata
    run_name = cfg.inference.run_name or f"agent_{get_cool_name()}"
    run_name = f"{cfg.inference.run_prefix}{run_name}{cfg.inference.run_suffix}"
    tags = [
        "agent",
        data_source.name.split("/")[1],
        type(llm).__name__,
        *tool_factory.short_names,  # all the tool names
    ]
    metadata = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # Agent code editor
    code_editor = AgentCodeEditor(
        agent_graph_partial=agent_graph_partial,
        tool_factory=tool_factory,
        data_path=data_path,
        context_providers_cfg=cfg.context,
        runnable_config=RunnableConfig(run_name=run_name, tags=tags, metadata=metadata, recursion_limit=1024),
    )

    # Name for this run
    codeeditor_name = f"{type(llm).__name__}/{tool_factory.short_name}"

    # Output path
    output_path = init_output_path(codeeditor_name, cfg, data_source)

    # WandB initialization
    init_wandb(cfg, run_name, tags=tags)

    # Perform inference loop and save predictions
    with logging_redirect_tqdm():
        inference_loop(code_editor, data_source, output_path, cfg.inference, run_name)
    finish_wandb()


if __name__ == "__main__":
    main()
