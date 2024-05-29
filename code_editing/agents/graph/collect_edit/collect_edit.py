from typing import Dict, List

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from code_editing.agents.graph.graph_factory import AgentInput, GraphFactory


class CollectEditState(AgentInput):
    collected_context: Dict[str, List[int]]


class CollectEdit(GraphFactory):
    name = "collect_edit"

    def __init__(self, context_collector: GraphFactory, editor: GraphFactory, only_collect: bool = False):
        super().__init__()
        self.context_collector = context_collector
        self.editor = editor
        self.only_collect = only_collect

    def build(self, *args, retrieval_helper=None, **kwargs):
        self.context_collector.copy_from(self, copy_tools=True)
        self.editor.copy_from(self, copy_tools=True)

        workflow = StateGraph(dict)

        def validate_intermediate(state: dict):
            if "collected_context" not in state:
                raise ValueError("`collected_context` key is not in the state. Please check the workflow.")
            for key, value in state["collected_context"].items():
                if 0 in list(value):
                    raise ValueError(f"Lines in {key} are 0-indexed. Please check the workflow.")
            return state

        workflow.add_node("collect", self.context_collector.build(retrieval_helper=retrieval_helper))
        workflow.add_node("validate", RunnableLambda(validate_intermediate, name="validate"))
        workflow.add_edge("collect", "validate")

        if not self.only_collect:
            workflow.add_node("edit", self.editor.build(retrieval_helper=retrieval_helper))
            workflow.add_edge("edit", END)
        workflow.add_edge("validate", "edit" if not self.only_collect else END)

        workflow.set_entry_point("collect")

        return workflow.compile()
