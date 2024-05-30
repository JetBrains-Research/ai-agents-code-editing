import logging
from typing import Dict, List, Union

from hydra.utils import get_class, instantiate
from langchain_core.runnables import RunnableConfig

from code_editing.agents.collect_edit.context_collectors.acr_search.search_manage import SearchManager
from code_editing.agents.graph_factory import GraphFactory
from code_editing.agents.utils.checkout_extractor import CheckoutExtractor
from code_editing.agents.utils.tool_factory import ToolFactory
from code_editing.backbones.base_backbone import CEInput, CEOutput
from code_editing.baseline.baseline import CodeEditor
from code_editing.configs.agents.loader_config import LoaderConfig
from code_editing.configs.agents.retrieval_config import RetrievalConfig
from code_editing.utils.git_utils import get_head_diff_unsafe, reset_to_head
from code_editing.utils.wandb_utils import log_codeeditor_trace


class AgentCodeEditor(CodeEditor):
    def __init__(
        self,
        graph_factory: GraphFactory,
        tool_factory: ToolFactory,
        data_path: str,
        retrieval_cfg: Union[RetrievalConfig, Dict] = None,
        loader_cfg: Union[LoaderConfig, Dict] = None,
        tools_cfg: Dict = None,
        tags: List[str] = None,
        metadata: Dict = None,
        run_name: str = None,
    ):
        if retrieval_cfg is None:
            retrieval_cfg = {}
        if loader_cfg is None:
            loader_cfg = {}
        if tools_cfg is None:
            tools_cfg = {}
        if tags is None:
            tags = []
        if metadata is None:
            metadata = {}
        if run_name is None:
            run_name = "Agent"

        # Agent graph
        self.graph_factory = graph_factory
        # Tools and Data
        self.tool_factory = tool_factory
        self.tools_cfg = tools_cfg
        self.data_path = data_path
        # Retrieval
        self.retrieval_cfg = retrieval_cfg
        # Document loader
        self.loader_cls = get_class(loader_cfg["_target_"])
        self.loader_kwargs = dict(loader_cfg)
        self.loader_kwargs.pop("_target_")
        # Metadata for tracing
        self.tags = tags
        self._metadata = metadata
        self.run_name = run_name

    @log_codeeditor_trace()
    def generate_diff(self, req: CEInput, root_span) -> CEOutput:
        # Get repository full path
        repo_path = req["code_base"].get(CheckoutExtractor.REPO_KEY, None)
        if repo_path is None:
            raise ValueError("repo_path is required")

        retrieval_helper = instantiate(
            self.retrieval_cfg,
            repo_path=repo_path,
            loader_cls=self.loader_cls,
            loader_kwargs=self.loader_kwargs,
            data_path=self.data_path,
        )
        search_manager = SearchManager(project_path=repo_path)
        search_manager.show_lineno = True

        tools = self.tool_factory.build(
            data_path=self.data_path,
            repo_path=repo_path,
            retrieval_helper=retrieval_helper,
            search_manager=search_manager,
            root_span=root_span,  # W&B root span
            **self.tools_cfg,
        )

        # Build the graph runnable
        app = self.graph_factory.tools(tools).build(retrieval_helper=retrieval_helper)

        # Invoke the graph
        res = app.invoke(
            input={"instruction": req["instruction"]},
            config=RunnableConfig(
                tags=self.tags,
                metadata=self._metadata,
                run_name=self.run_name,
                recursion_limit=1024,
            ),
        )
        # Get diff
        diff = get_head_diff_unsafe(repo_path, self.data_path)
        # Get lines viewed
        viewed_lines = res.get("collected_context", None)
        if viewed_lines is None:
            logging.warning("No viewed lines found in the graph output")
            viewed_lines = retrieval_helper.viewed_lines
        # Reset the repository to the head commit
        reset_to_head(repo_path, self.data_path)

        return {"prediction": diff, "viewed_lines": viewed_lines}

    @property
    def metadata(self) -> dict:
        return {
            "type": "agent",
            "graph_factory": self.graph_factory.name,
        }
