from typing import Any

from pydantic import BaseModel, Field

from code_editing.agents.context_providers.retrieval.retrieval_helper import RetrievalHelper
from code_editing.agents.tools.base_tool import CEBaseTool
from code_editing.agents.tools.common import my_format_fragment, parse_file, read_file
from code_editing.code_editor import CEBackbone


class EditTool(CEBaseTool):
    class EditToolInput(BaseModel):
        file_name: str = Field(description="File name to edit", examples=["main.py", "test/benchmark/metrics.py"])
        start_index: int = Field(description="Start index of the fragment to edit", examples=[0, 100])
        instruction: str = Field(
            description="Instruction for the editing LLM", examples=["Replace the for loop with a while loop"]
        )
        context: int = Field(
            description="Number of lines (before and after) added to the fragment to be edited",
            default=5,
            examples=[5, 10],
        )

    name = "edit-fragment"
    description = """Edit a fragment of code based on the given instruction.
    Useful to make changes to the code base. Inputs are the file name, the start index, the instruction and the context size.
    The instruction should be a prompt for the editing LLM."""
    args_schema = EditToolInput

    def __init__(self, backbone: CEBackbone = None, root_span=None, **kwargs):
        super().__init__(**kwargs)
        self.args_schema = self.EditToolInput

        if self.dry_run:
            return

        self.backbone = backbone
        self.root_span = root_span

        if self.backbone is None:
            raise ValueError("Backbone is required for the edit tool")

        # noinspection PyTypeChecker
        self.retrieval_helper = self.run_overview_manager.get_ctx_provider("retrieval_helper")

    def _run_tool(self, file_name: str, start_index: int, instruction: str, context: int = 5) -> Any:
        start_index = int(start_index)
        file = parse_file(file_name, self.repo_path)
        contents, lines, start, end = read_file(context, file, start_index)
        # Send to the editing LLM
        resp = self.backbone.generate_diff(
            {"instruction": instruction, "code_base": {file_name: contents}}, parent_span=self.root_span
        )
        new_contents = resp["prediction"]
        # Save
        with open(file, "w") as f:
            f.write("".join(lines[:start]) + new_contents + "".join(lines[end:]))
        # Reindex
        self.retrieval_helper.add_changed_file(file)
        # Return the new fragment
        result = my_format_fragment(source=file_name, start_index=start_index, page_content=new_contents)
        return "Code has been updated. Here is the new fragment:\n" + result

    @property
    def short_name(self) -> str:
        return f"edit"

    backbone: CEBackbone = None
    retrieval_helper: RetrievalHelper = None
    root_span: Any = None
