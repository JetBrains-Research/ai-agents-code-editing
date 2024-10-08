import logging
from typing import Callable, Dict

from hydra.utils import instantiate
from langchain_core.runnables import RunnableConfig, RunnableLambda

from code_editing.agents.agent_graph import AgentGraphPartial
from code_editing.agents.run import AgentRunManager
from code_editing.agents.utils.checkout_extractor import CheckoutExtractor
from code_editing.agents.utils.tool_factory import ToolFactory
from code_editing.code_editor import CEInput, CEOutput, CodeEditor
from code_editing.configs.agents.context_providers.context_config import ContextConfig
from code_editing.utils.file_log import MyFileCallbackHandler
from code_editing.utils.git_utils import get_head_diff_unsafe


class AgentCodeEditor(CodeEditor):
    def __init__(
        self,
        agent_graph_partial: AgentGraphPartial,
        tool_factory: ToolFactory,
        data_path: str,
        context_providers_cfg: Dict[str, ContextConfig] = None,
        runnable_config: RunnableConfig = None,
    ):
        if context_providers_cfg is None:
            context_providers_cfg = {}

        # Agent graph
        self.agent_graph_partial = agent_graph_partial
        # Tools and Data
        self.tool_factory = tool_factory
        self.data_path = data_path

        self.context_providers_cfg = context_providers_cfg
        self.runnable_config = runnable_config

    def generate_diff(self, req: CEInput) -> CEOutput:
        # Get repository full path
        repo_path = req["code_base"].get(CheckoutExtractor.REPO_KEY, None)
        if repo_path is None:
            raise ValueError("repo_path is required")

        generation_kwargs = {"repo_path": repo_path, "data_path": self.data_path}

        # Context providers that help the agent to search for the code
        context_providers = {k: instantiate(v, **generation_kwargs) for k, v in self.context_providers_cfg.items()}

        run_manager = AgentRunManager(
            **generation_kwargs,
            context_providers=context_providers,
            instance_id=req.get("instance_id", None),
            raw_data=req.get("raw_data", None),
        )

        # Tools available to the agent
        tools = self.tool_factory.build(
            run_manager=run_manager,
        )

        app = self.agent_graph_partial(tools=tools, run_manager=run_manager)

        # Diff collection
        def to_ceoutput(state):
            diff = get_head_diff_unsafe(repo_path, self.data_path)
            # Get lines viewed
            viewed_lines = state.get("collected_context", None)
            if viewed_lines is None:
                logging.warning("No viewed lines found in the graph output")
                viewed_lines = {}
            return {
                "raw": state,
                "prediction": diff,
                "viewed_lines": viewed_lines,
                "run": run_manager.get_run_summary(),
            }

        # update runnable config
        runnable_config = self.runnable_config.copy()
        runnable_config["run_name"] = f"{runnable_config['run_name']}.{run_manager.instance_id}"
        runnable_config.setdefault("callbacks", [])
        # noinspection PyTypeChecker
        runnable_config["callbacks"].append(MyFileCallbackHandler(run_manager.get_log_path()))

        # Invoke the graph
        return (app | RunnableLambda(to_ceoutput, name="Collect Diff")).invoke(
            input={"instruction": req["instruction"]},
            config=runnable_config,
        )
