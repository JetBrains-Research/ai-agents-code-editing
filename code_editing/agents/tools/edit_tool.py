from pydantic import BaseModel, Field

from code_editing.agents.context_providers.retrieval.retrieval_helper import RetrievalHelper
from code_editing.agents.tools.base_tool import CEBaseTool
from code_editing.agents.tools.common import parse_file, read_file_full, read_file_lines
from code_editing.code_editor import CEBackbone


class EditTool(CEBaseTool):
    class EditToolInput(BaseModel):
        file_name: str = Field(description="File name to edit", examples=["main.py", "test/benchmark/metrics.py"])
        to_replace: str = Field(description="The code to replace", examples=["def old_function():\n    pass\n"])
        new_code: str = Field(description="The new code", examples=["def new_function():\n    pass\n"])

    name = "edit-fragment"
    description = """Edit a fragment of code. Inputs are the old code to replace and the new code to replace it with."""
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

    def _run_tool(self, file_name: str, to_replace: str, new_code: str) -> str:
        file = parse_file(file_name, self.repo_path)
        # Read the file
        contents = read_file_full(file)
        # Find the fragment to replace
        start = contents.find(to_replace)
        start_line = contents.count("\n", 0, start)
        if start == -1:
            return "The code to replace was not found in the file."
        end = start + len(to_replace)
        # Replace the fragment
        new_contents = contents[:start] + new_code + contents[end:]
        # Save
        with open(file, "w") as f:
            f.write(new_contents)
        # Reindex
        self.retrieval_helper.add_changed_file(file)
        # Return the new fragment
        new_state = read_file_lines(file, start_line - 5, start_line + new_code.count("\n") + 1 + 5)[0]

        return "Code has been updated. Here is the context around the change:\n" + new_state

    @property
    def short_name(self) -> str:
        return f"edit"

    backbone: CEBackbone = None
    retrieval_helper: RetrievalHelper = None
