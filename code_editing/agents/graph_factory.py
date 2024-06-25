import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict

import hydra.utils
from langchain.agents import AgentExecutor
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from typing_extensions import Self

from code_editing.agents.run import RunOverviewManager
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

    def __init__(
        self,
        chat_prompt=None,
        agent_executor_cfg=None,
        llm=None,
        create_agent_method=None,
    ):
        self._tools = []
        self._chat_prompt = chat_prompt
        self._agent_executor_cfg = agent_executor_cfg or {}
        self._llm: Optional[BaseChatModel] = llm
        self._create_agent_method = create_agent_method

    @abstractmethod
    def build(self, run_overview_manager: RunOverviewManager, *args, **kwargs) -> Runnable[AgentInput, Any]:
        pass

    # Utility functions for the derived classes

    def _agent_executor(self, **kwargs) -> AgentExecutor:
        tools = self._tools
        if "tools" in kwargs:
            tools = kwargs.pop("tools")
        if not tools:
            tools = [dummy]
        agent = hydra.utils.call(
            {"_target_": self._create_agent_method}, llm=self._llm, tools=tools, prompt=self._chat_prompt
        )
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

    def create_agent_method(self, create_agent_method: str) -> Self:
        self._create_agent_method = create_agent_method
        return self

    def copy_from(self, other: "GraphFactory", copy_tools: bool = True) -> Self:
        if copy_tools:
            self._tools = other._tools
        self._chat_prompt = other._chat_prompt
        self._agent_executor_cfg = other._agent_executor_cfg
        self._llm = other._llm
        self._create_agent_method = other._create_agent_method
        return self

    def get_logger(self, run_overview_manager: RunOverviewManager):
        return logging.getLogger(f"inference.{run_overview_manager.instance_id}.{self.name}")
