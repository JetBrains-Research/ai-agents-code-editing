from pydantic import BaseModel, Field

from code_editing.agents.tools.base_tool import CEBaseTool
from code_editing.agents.tools.common import my_format_fragment, parse_file, read_file


class ViewFileTool(CEBaseTool):
    class ViewFileToolInput(BaseModel):
        file_name: str = Field(description="File name to view", examples=["main.py", "test/benchmark/metrics.py"])
        start_index: int = Field(description="Start index of the fragment to view", examples=[0, 100])
        context: int = Field(
            description="Number of lines (before and after) added to the fragment to be viewed", examples=[5, 10]
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args_schema = self.ViewFileToolInput
        if self.dry_run:
            return

    name = "view-fragment"
    description = """View a file from the code base. Useful to see the context of a index.
    Inputs are the file name, the start index and the context size."""
    args_schema = ViewFileToolInput

    def _run_tool(self, file_name: str, start_index: int, context: int):
        start_index = int(start_index)
        file = parse_file(file_name, self.repo_path)
        contents, lines, start, end = read_file(context, file, start_index)
        return my_format_fragment(source=file_name, start_index=start_index, page_content=contents)

    @property
    def short_name(self) -> str:
        return "vf"
