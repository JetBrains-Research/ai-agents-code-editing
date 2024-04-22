from langchain_core.tools import tool, ToolException

from code_editing.agents.graph.graph_factory import GraphFactory
from code_editing.agents.tools.common import parse_file, read_file_full
from code_editing.agents.utils import PromptWrapper


class LLMRetrieval(GraphFactory):
    name = "llm_retrieval"

    def __init__(self, search_prompt: PromptWrapper, do_review: bool = True, **kwargs):
        super().__init__()
        self.search_prompt = search_prompt
        self.do_review = do_review

    def build(self, *args, retrieval_helper=None, **kwargs):
        if retrieval_helper is None:
            raise ValueError("Retrieval helper is not set")

        return (
            self.search_prompt.as_runnable(to_dict=True)
            | self._agent_executor(tools=self.get_llm_retrieval_tools(retrieval_helper))
            | {"collected_context": lambda _: retrieval_helper.viewed_lines}
        )

    def get_llm_retrieval_tools(self, retrieval_helper):
        # Find the code search tool
        search_tools = [t for t in self._tools if "search" in t.name]

        @tool
        def add_to_context(file_name: str, start_line: int, end_line: int):
            """Adds the code snippet to the context for the subsequent editing.

            Accepts the file name, start line and end line (exclusive) of the code snippet.
            """
            try:
                file = parse_file(file_name, retrieval_helper.repo_path)
                contents = read_file_full(file)
                num_lines = len(contents.split("\n"))
            except ToolException as e:
                return str(e)
            if start_line <= 0:
                return "Start line must be a positive integer"
            if end_line <= start_line:
                return "End line must be greater than the start line"
            if end_line > num_lines:
                return f"End line is greater than the number of lines in the file ({num_lines})"
            retrieval_helper.add_viewed_doc(file_name, start_line, end_line)
            return (
                "Code snippet has been added to the context. You can add more snippets or finish the run if you "
                "think you have found all relevant code."
            )

        tools = search_tools
        if self.do_review:
            tools.append(add_to_context)
            # for code_search_tool in search_tools:
            #     code_search_tool.do_add_to_viewed = False  # noqa

        return tools
