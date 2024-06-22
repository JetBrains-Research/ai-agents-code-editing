import logging
import os.path
from operator import itemgetter

from langchain_core.runnables import RunnableLambda

from code_editing.agents.collect_edit.editors.util import TagParser
from code_editing.agents.context_providers.aider import AiderRepoMap
from code_editing.agents.graph_factory import GraphFactory
from code_editing.agents.run import RunOverviewManager
from code_editing.agents.tools.common import read_file_full
from code_editing.agents.utils import PromptWrapper

logger = logging.getLogger(__name__)


class AiderRetrieval(GraphFactory):
    name = "aider_retrieval"

    def __init__(self, select_prompt: PromptWrapper, **kwargs):
        super().__init__()
        self.select_prompt = select_prompt

    def build(self, run_overview_manager: RunOverviewManager, *args, **kwargs):
        # noinspection PyTypeChecker
        aider: AiderRepoMap = run_overview_manager.get_ctx_provider("aider")
        repo_map = aider.get_repo_map()

        def to_viewed_lines(state: dict):
            files = set(state["matches"])
            viewed_lines = {}
            for file in files:
                full_path = os.path.join(aider.repo_path, file)
                if not os.path.exists(full_path):
                    logger.warning(f"File {full_path} does not exist")
                    continue
                line_cnt = len(read_file_full(full_path).split("\n"))
                viewed_lines[file] = list(range(1, line_cnt + 1))
            return {"collected_context": viewed_lines}

        return (
            {"repo_map": lambda _: repo_map, "instruction": itemgetter("instruction")}
            | self.select_prompt.as_runnable()
            | self._llm
            | TagParser(tag="file")
            | RunnableLambda(to_viewed_lines, name="Convert to viewed lines")
        )
