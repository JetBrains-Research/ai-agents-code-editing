from pydantic import BaseModel, Field

from code_editing.agents.tools.base_tool import CEBaseTool
from code_editing.agents.tools.common import parse_file, read_file_lines


class ViewFileTool(CEBaseTool):
    class ViewFileToolInput(BaseModel):
        file_name: str = Field(description="File name to view", examples=["main.py", "test/benchmark/metrics.py"])
        line_start_number: int = Field(description="Line start number to view", examples=[1, 100])
        line_end_number: int = Field(description="Line end number to view", examples=[10, 120])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args_schema = self.ViewFileToolInput
        if self.dry_run:
            return

    name = "view-fragment"
    description = """View a fragment of code based on the given line numbers. Useful to view the code base. 1-based
    line numbers are used."""
    args_schema = ViewFileToolInput

    def _run_tool(self, file_name: str, line_start_number: int, line_end_number: int) -> str:
        line_start_number, line_end_number = int(line_start_number), int(line_end_number)
        file = parse_file(file_name, self.repo_path)
        contents, line_start, line_end, _ = read_file_lines(file, line_start_number, line_end_number, True)
        return f"+++ {file_name}\n{contents}"

    @property
    def short_name(self) -> str:
        return "vf"
