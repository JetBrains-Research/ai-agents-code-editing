from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from typing_extensions import Self

from code_editing.agents.tools.common import dummy


class AgentInput(TypedDict):
    """This is a TypedDict class that represents the input to the agent."""

    instruction: str


class GraphFactory(ABC):
    """
    This is an abstract class that represents a factory for creating a langgraph runnable for code editing.

    The runnable accepts an input of type AgentInput.
    """

    name = "base"

    def __init__(self):
        self._tools = []
        self._chat_prompt = None
        self._agent_executor_cfg = {}
        self._llm: Optional[BaseChatModel] = None

    @abstractmethod
    def build(self, *args, **kwargs) -> Runnable[AgentInput, Any]:
        pass

    # Utility functions for the derived classes

    def _agent_executor(self, **kwargs) -> AgentExecutor:
        tools = self._tools
        if "tools" in kwargs:
            tools = kwargs.pop("tools")
        if not tools:
            tools = [dummy]
        agent = create_openai_tools_agent(self._llm, tools, self._chat_prompt)
        return AgentExecutor(
            tools=tools,
            agent=agent,
            **self._agent_executor_cfg,
            **kwargs,
        )

    # Setters

    def tools(self, tools: List[BaseTool]) -> Self:
        self._tools = tools
        return self

    def chat_prompt(self, chat_prompt) -> Self:
        self._chat_prompt = chat_prompt
        return self

    def agent_executor_cfg(self, agent_executor_cfg: Dict) -> Self:
        self._agent_executor_cfg = agent_executor_cfg
        return self

    def llm(self, llm: BaseLLM) -> Self:
        self._llm = llm
        return self

    def copy_from(self, other: "GraphFactory", copy_tools: bool = True) -> Self:
        if copy_tools:
            self._tools = other._tools
        self._chat_prompt = other._chat_prompt
        self._agent_executor_cfg = other._agent_executor_cfg
        self._llm = other._llm
        return self
