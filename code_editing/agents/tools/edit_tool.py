from pydantic import BaseModel, Field

from code_editing.agents.context_providers.retrieval.retrieval_helper import RetrievalHelper
from code_editing.agents.tools.base_tool import CEBaseTool
from code_editing.agents.tools.common import parse_file, read_file_lines
from code_editing.code_editor import CEBackbone


class EditTool(CEBaseTool):
    class EditToolInput(BaseModel):
        file_name: str = Field(description="File name to edit", examples=["main.py", "test/benchmark/metrics.py"])
        line_start_number: int = Field(description="Line start number to view", examples=[1, 100])
        line_end_number: int = Field(description="Line end number to view", examples=[10, 120])
        new_contents: str = Field(description="New contents of the file", examples=["def new_function():\n    pass\n"])

    name = "edit-fragment"
    # description = """Edit a fragment of code based on the given instruction.
    # Useful to make changes to the code base. Inputs are the file name, the start index, the instruction and the context size.
    # The instruction should be a prompt for the editing LLM."""
    description = """Edit a fragment of code. Inputs are the file name, the start index, and the new code for the
    fragment."""
    args_schema = EditToolInput

    def __init__(self, backbone: CEBackbone = None, **kwargs):
        super().__init__(**kwargs)
        self.args_schema = self.EditToolInput

        if self.dry_run:
            return

        self.backbone = backbone

        if self.backbone is None:
            raise ValueError("Backbone is required for the edit tool")

        # noinspection PyTypeChecker
        self.retrieval_helper = self.run_overview_manager.get_ctx_provider("retrieval_helper")

    def _run_tool(self, file_name: str, line_start_number: int, line_end_number: int, new_contents: str) -> str:
        line_start_number, line_end_number = int(line_start_number), int(line_end_number)
        file = parse_file(file_name, self.repo_path)
        contents, start, end, lines = read_file_lines(file, line_start_number, line_end_number)
        # Send to the editing LLM
        # resp = self.backbone.generate_diff({"instruction": instruction, "code_base": {file_name: contents}})
        # new_contents = resp["prediction"]
        # Save
        with open(file, "w") as f:
            f.write("\n".join(lines[:start - 1] + [new_contents] + lines[end:]))
        # Reindex
        self.retrieval_helper.add_changed_file(file)
        # Return the new fragment
        # result = my_format_fragment(source=file_name, start_index=start_index, page_content=new_contents)
        new_state, *_ = read_file_lines(file, start - 5, start + new_contents.count("\n") + 5, True)
        return "Code has been updated. Here is the context around the change:\n" + new_state

    @property
    def short_name(self) -> str:
        return f"edit"

    backbone: CEBackbone = None
    retrieval_helper: RetrievalHelper = None
