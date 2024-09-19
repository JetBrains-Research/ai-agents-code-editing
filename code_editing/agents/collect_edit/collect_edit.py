from typing import Dict, List

from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph import END, StateGraph

from code_editing.agents.agent_graph import AgentGraph, AgentGraphPartial, AgentInput


class CollectEditState(AgentInput):
    collected_context: Dict[str, List[int]]


class CollectEdit(AgentGraph):
    name = "collect_edit"

    def __init__(
        self, context_collector: AgentGraphPartial, editor: AgentGraphPartial, only_collect: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.context_collector = context_collector
        self.editor = editor
        self.only_collect = only_collect

    @property
    def _runnable(self) -> Runnable:
        context_collector = self.context_collector(**self.root_params)
        editor = self.editor(**self.root_params)

        workflow = StateGraph(dict)

        def validate_intermediate(state: dict):
            if "collected_context" not in state:
                raise ValueError("`collected_context` key is not in the state. Please check the workflow.")
            for key, value in state["collected_context"].items():
                if 0 in list(value):
                    raise ValueError(f"Lines in {key} are 0-indexed. Please check the workflow.")
            return state

        workflow.add_node("collect", context_collector)
        workflow.add_node("validate", RunnableLambda(validate_intermediate, name="validate"))
        workflow.add_edge("collect", "validate")

        if not self.only_collect:
            workflow.add_node("edit", editor)
            workflow.add_edge("edit", END)
        workflow.add_edge("validate", "edit" if not self.only_collect else END)

        workflow.set_entry_point("collect")

        return workflow.compile()
