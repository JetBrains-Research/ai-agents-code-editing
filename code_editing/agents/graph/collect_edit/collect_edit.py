from typing import Dict, List

from langgraph.graph import StateGraph, END

from code_editing.agents.graph.graph_factory import GraphFactory, AgentInput


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

        workflow.add_node("collect", self.context_collector.build(retrieval_helper=retrieval_helper))

        if not self.only_collect:
            workflow.add_node("edit", self.editor.build(retrieval_helper=retrieval_helper))
            workflow.add_edge("edit", END)
        workflow.add_edge("collect", "edit" if not self.only_collect else END)

        workflow.set_entry_point("collect")

        return workflow.compile()
