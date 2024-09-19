from operator import itemgetter

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from code_editing.agents.agent_graph import AgentGraph, AgentInput
from code_editing.agents.utils.user_prompt import PromptWrapper


class SelfReflectionState(AgentInput):
    review: str
    should_continue: bool
    intermediate_steps: list


class SelfReflection(AgentGraph):
    """
    Langgraph self-reflection system for code editing.

    This system includes an agent that edits code and a reviewer that reviews the agent's edits.
    """

    name = "self_reflection"

    def __init__(
        self, agent_prompt: PromptWrapper, agent_review_prompt: PromptWrapper, review_prompt: PromptWrapper, **kwargs
    ):
        super().__init__(**kwargs)
        self.agent_prompt = agent_prompt
        self.agent_review_prompt = agent_review_prompt
        self.review_prompt = review_prompt

    @property
    def _runnable(self, *args, **kwargs):
        raise NotImplementedError()
        memory = InMemoryChatMessageHistory()
        agent_executor = self.agent(memory=memory)

        def agent_init_step(inp: AgentInput) -> SelfReflectionState:
            """First agent step (without review)"""
            agent_init = self.agent_prompt.as_runnable(to_dict=True) | agent_executor
            res = agent_init.invoke(inp)
            return {
                "instruction": inp["instruction"],
                "review": "",
                "should_continue": True,
                "intermediate_steps": res["intermediate_steps"],
            }

        def review_step(state: SelfReflectionState) -> SelfReflectionState:
            """Review the agent's actions."""
            # Define the output parser
            response_schemas = [
                ResponseSchema(name="review", description="Text review of the AI's actions.", type="str"),
                ResponseSchema(
                    name="should_continue",
                    description="true if the AI should edit the code again, false otherwise.",
                    type="bool",
                ),
            ]
            output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
            # Reviewer runnable
            agent_review = self.review_prompt.as_runnable() | self._llm | output_parser

            # Run the reviewer
            inputs = state.copy()
            inputs["format_instructions"] = output_parser.get_format_instructions()
            inputs["chat_history"] = str(memory.messages)
            inputs["intermediate_steps"] = str(state["intermediate_steps"])
            res = agent_review.invoke(inputs)
            state["review"] = res["review"]
            state["should_continue"] = res["should_continue"]
            return state

        def agent_cont_step(state: SelfReflectionState) -> SelfReflectionState:
            """Agent rerun with review."""
            agent_cont = self.agent_review_prompt.as_runnable(to_dict=True) | agent_executor

            res = agent_cont.invoke(state)
            state["intermediate_steps"].extend(res["intermediate_steps"])

            return state

        workflow = StateGraph(SelfReflectionState)
        workflow.add_node("agent_init", RunnableLambda(agent_init_step))
        workflow.add_node("agent_cont", RunnableLambda(agent_cont_step))
        workflow.add_node("reviewer", RunnableLambda(review_step))

        workflow.set_entry_point("agent_init")

        workflow.add_edge("agent_init", "reviewer")
        workflow.add_conditional_edges("reviewer", itemgetter("should_continue"), {True: "agent_cont", False: END})
        workflow.add_edge("agent_cont", "reviewer")

        return workflow.compile()
