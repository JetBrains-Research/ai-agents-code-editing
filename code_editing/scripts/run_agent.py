import dotenv
import hydra
import omegaconf
from hydra.utils import instantiate
from langchain_core.language_models import BaseLanguageModel
from tqdm.contrib.logging import logging_redirect_tqdm

dotenv.load_dotenv()

from code_editing.agents.agent_codeeditor import AgentCodeEditor
from code_editing.agents.graph_factory import GraphFactory
from code_editing.agents.utils.chat_prompt import ChatPromptFactory
from code_editing.agents.utils.checkout_extractor import CheckoutExtractor
from code_editing.agents.utils.tool_factory import ToolFactory
from code_editing.code_editor import CEBackbone
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
    prompt_factory: ChatPromptFactory = instantiate(cfg.chat_prompt)
    llm: BaseLanguageModel = instantiate(cfg.llm)
    backbone: CEBackbone = instantiate(cfg.backbone)

    chat_prompt = prompt_factory.build()

    # Set up the graph factory for the agent interactions
    graph_factory: GraphFactory = (
        instantiate(cfg.graph).chat_prompt(chat_prompt).llm(llm).agent_executor_cfg(cfg.agent_executor)
    )

    # Set up the tracing tags and metadata
    cool_name = get_cool_name()
    tags = [
        "agent",
        data_source.name.split("/")[1],
        graph_factory.name,
        prompt_factory.name,
        type(llm).__name__,
        *tool_factory.short_names,  # all the tool names
    ]
    metadata = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    run_name = f"Agent {cool_name}"

    # Agent code editor
    code_editor = AgentCodeEditor(
        graph_factory=graph_factory,
        tool_factory=tool_factory,
        data_path=data_path,
        retrieval_cfg=cfg.retrieval,
        loader_cfg=cfg.loader,
        tools_cfg={
            "handle_tool_error": True,
            "backbone": backbone,
        },
        tags=tags,
        metadata=metadata,
        run_name=run_name,
    )

    # Name for this run
    codeeditor_name = f"{prompt_factory.name}/{type(llm).__name__}/{graph_factory.name}_bck-{backbone.name.replace('/', '-')}_{tool_factory.short_name}"

    # Output path
    output_path = init_output_path(codeeditor_name, cfg, data_source)

    # WandB initialization
    init_wandb(cfg, run_name, tags=tags)

    # Perform inference loop and save predictions
    with logging_redirect_tqdm():
        inference_loop(
            code_editor,
            data_source,
            output_path,
            cfg.inference,
        )
    finish_wandb()


if __name__ == "__main__":
    main()
