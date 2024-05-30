from typing import Any

from langchain_core.runnables import Runnable

from code_editing.agents.graph_factory import AgentInput, GraphFactory
from code_editing.agents.utils.user_prompt import PromptWrapper


class AgentOnly(GraphFactory):
    """
    Code editing agent-only system for code editing.

    This system includes an agent that edits code.
    """

    name = "agent_only"

    def __init__(self, agent_prompt: PromptWrapper, **kwargs):
        super().__init__()
        self.agent_prompt = agent_prompt

    def build(self, *args, **kwargs) -> Runnable[AgentInput, Any]:
        return self.agent_prompt.as_runnable(to_dict=True) | self._agent_executor()
