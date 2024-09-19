from langchain_core.runnables import Runnable

from code_editing.agents.agent_graph import AgentGraph
from code_editing.agents.utils.user_prompt import PromptWrapper


class AgentOnly(AgentGraph):
    name = "agent_only"

    def __init__(self, agent_prompt: PromptWrapper, **kwargs):
        super().__init__(**kwargs)
        self.agent_prompt = agent_prompt

    @property
    def _runnable(self) -> Runnable:
        return self.agent_prompt.as_runnable(to_dict=True) | self.react_agent()
