from operator import itemgetter
from typing import List

from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.output_parsers import OutputFixingParser
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langgraph.graph import StateGraph, END

from code_editing.agents.graph.collect_edit.collect_edit import CollectEditState
from code_editing.agents.graph.collect_edit.editors.util import process_edit, MarkdownOutputParser
from code_editing.agents.graph.graph_factory import GraphFactory
from code_editing.agents.tools.common import write_file_full, parse_file
from code_editing.agents.utils import PromptWrapper


class EditorState(CollectEditState):
    edited_files: List[str]


class SimpleEditor(GraphFactory):
    def __init__(self, edit_prompt: PromptWrapper):
        super().__init__()
        self.edit_prompt = edit_prompt

    def build(self, *args, retrieval_helper=None, **kwargs):
        workflow = StateGraph(EditorState)

        agent_executor = self._agent_executor(
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output"),
            tools=[]
        )

        def edit(state: EditorState) -> EditorState:
            collected_context = state["collected_context"]
            # Find first file that has not been edited
            keys = set(collected_context.keys())
            edited_files = set(state["edited_files"] or [])
            if len(keys - edited_files) == 0:
                return state
            file_name = next(iter(keys - edited_files))
            # Edit the file
            lines = collected_context[file_name]

            def edit_lambda(_: str, snippet: str) -> str:
                inp = {"file_name": file_name, "code": snippet, "instruction": state["instruction"]}
                parser = MarkdownOutputParser(key="editedcode")
                fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=self._llm)
                self.edit_prompt.overrides["format_instructions"] = parser.get_format_instructions()
                return RunnableSequence(
                    self.edit_prompt.as_runnable(to_dict=True),
                    agent_executor,
                    itemgetter("output") | fixing_parser,
                    name=f"Edit {file_name}",
                ).invoke(inp)

            file = parse_file(file_name, retrieval_helper.repo_path)
            new_code = process_edit(file, lines, edit_lambda)
            # Write the new code to the file
            write_file_full(file, new_code)
            # Update the state
            res = state.copy()
            res["edited_files"] = list(edited_files | {file_name})
            return res

        workflow.add_node("file_edit", RunnableLambda(edit, name="File Edit"))

        def is_not_done(state: EditorState) -> bool:
            collected = state.get("collected_context", {}) or {}
            edited = state.get("edited_files", []) or []
            return len(set(collected.keys()) - set(edited)) > 0

        workflow.add_conditional_edges(
            "file_edit",
            is_not_done,
            {True: "file_edit", False: END},
        )
        workflow.set_entry_point("file_edit")

        return workflow.compile()
