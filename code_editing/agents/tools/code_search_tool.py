from typing import Optional, Any

from pydantic import BaseModel, Field

from code_editing.agents.tools.base_tool import CEBaseTool
from code_editing.agents.tools.common import lines_format_document


class CodeSearchTool(CEBaseTool):
    def __init__(self, show_outputs: bool = True, calls_limit: Optional[int] = None, do_add_to_viewed: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.args_schema = self.CodeSearchToolInput
        if self.dry_run:
            return
        self.show_outputs = show_outputs
        self.calls_limit = calls_limit
        self.counter = 0
        self.do_add_to_viewed = do_add_to_viewed

        if self.retrieval_helper is None:
            raise ValueError("Retrieval helper is required for code search tool")

    class CodeSearchToolInput(BaseModel):
        query: str = Field(description="Search query. Should be a substring of the code you are looking for.", examples=["redundancy check", "get_user_by_id"])
        limit: int = Field(description="Number of search results to return", default=10)
        offset: int = Field(description="Offset for the search results", default=0, examples=[0, 10])

    name = "code-search"
    description = """Utility to search the code base. Useful to find the relevant code. Input should be a query,
    limit and offset.

    Your query should be a substring of the code you are looking for. The tool will return the code snippets that contain
    the substring.

    Do not use the tool to use semantic search. The tool is based on the substring search.
    """
    args_schema = CodeSearchToolInput

    def _run(self, query: str, limit: int = 10, offset: int = 0, run_manager=None) -> str:
        # Check the calls limit
        if self.calls_limit is not None and self.counter >= self.calls_limit:
            return "Code search calls limit reached. You can finish the run. Please, do not run the tool again."
        self.counter += 1

        # Search the vector store
        docs = self.retrieval_helper.search(query, offset + limit, run_manager)[offset:]
        # Register the viewed docs for the localization evaluation
        if self.do_add_to_viewed:
            self.retrieval_helper.add_viewed_docs(docs)
        # Format the search results for the agent to read
        res = self.document_separator.join(
            lines_format_document(doc, repo_path=self.retrieval_helper.repo_path) for doc in docs
        )
        if self.show_outputs:
            return res
        else:
            return "Code search has been executed successfully."

    @property
    def short_name(self) -> Optional[str]:
        return "cs"

    document_separator: str = "\n\n"
    document_prompt: Any = None
    show_outputs: bool = True
    calls_limit: Optional[int] = None
    counter: int = 0
    do_add_to_viewed: bool = True
