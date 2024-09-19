from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Type, TypedDict, TypeVar, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from code_editing.agents.context_providers.context_provider import ContextProvider
from code_editing.agents.run import AgentRunManager
from code_editing.agents.tools.common import dummy
from code_editing.scripts.common import logger


class AgentInput(TypedDict):
    """This is a TypedDict class that represents the input to the agent."""

    instruction: str


class AgentGraph(Runnable, ABC):
    """
    This is a helper class for making a langgraph runnable for code editing.

    The runnable accepts an input of type AgentInput.
    """

    name = "base_agentgraph"

    def __init__(
        self,
        tools: List[BaseTool] = None,
        llm: Optional[BaseChatModel] = None,
        run_manager: AgentRunManager = None,
        do_cache: bool = True,
    ):
        self.tools = tools
        self.llm = llm
        self.run_manager = run_manager
        self.do_cache = do_cache
        self.__cached_runnable: Optional[Runnable] = None

    @property
    @abstractmethod
    def _runnable(self) -> Runnable:
        pass

    def invoke(self, *args, **kwargs):
        if self.__cached_runnable is None or not self.do_cache:
            self.__cached_runnable = self._runnable
        return self.__cached_runnable.invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        if self.__cached_runnable is None or not self.do_cache:
            self.__cached_runnable = self._runnable
        return await self.__cached_runnable.ainvoke(*args, **kwargs)

    @property
    def root_params(self):
        return {"tools": self.tools, "llm": self.llm, "run_manager": self.run_manager, "do_cache": self.do_cache}

    # Utility functions for the derived classes
    def react_agent(self, **kwargs) -> Runnable:
        tools = self.tools
        if "tools" in kwargs:
            tools = kwargs.pop("tools")
        if not tools:
            logger.warning("No tools provided to the agent executor")
            tools = [dummy]

        def to_messages(state):
            if "messages" in state:
                return state
            if isinstance(state, str):
                return {"messages": [HumanMessage(state)]}
            if "messages" not in state and "input" in state:
                return {"messages": [HumanMessage(state["input"])]}
            raise ValueError(f"Invalid state: {state}")

        return (
            RunnableLambda(to_messages, name="to_messages")
            | create_react_agent(self.llm, tools)
            | RunnablePassthrough.assign(output=lambda x: x["messages"][-1].content)
        )

    T = TypeVar("T", bound=ContextProvider)

    def get_ctx_provider(self, ctx_provider_name: Union[str, Type[T]]) -> T:
        return self.run_manager.get_ctx_provider(ctx_provider_name)


AgentGraphPartial = Callable[..., AgentGraph]
