from operator import itemgetter

from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from code_editing.agents.collect_edit.context_collectors.llm_retrieval import LLMRetrieval
from code_editing.agents.context_providers.retrieval.retrieval_helper import RetrievalHelper
from code_editing.agents.run import AgentRunManager
from code_editing.agents.utils import PromptWrapper


class LLMCycleRetrieval(LLMRetrieval):
    name = "llm_cycle_retrieval"

    def __init__(self, review_prompt: PromptWrapper, max_tries: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.review_prompt = review_prompt
        self.max_tries = max_tries

    @property
    def _runnable(self):
        # FIXME
        raise NotImplementedError("This agent was broken by the langgraph refactor")
        retrieval_helper = self.get_ctx_provider(RetrievalHelper)

        agent = self.react_agent(tools=self.get_llm_retrieval_tools(retrieval_helper))

        workflow = StateGraph(dict)

        def search(state):
            out = (self.search_prompt.as_runnable(to_dict=True) | agent).invoke(state)
            res = state.copy()
            res["intermediate_steps"] = out["intermediate_steps"]
            return res

        review_counter = 0

        def review(state):
            nonlocal review_counter
            review_counter += 1

            output_parser = StructuredOutputParser.from_response_schemas(
                [
                    ResponseSchema(name="text_review", description="Review of the collected context", type="str"),
                    ResponseSchema(name="is_sufficient", description="Whether the context is sufficient", type="bool"),
                ]
            )
            inp = state.copy()
            inp["format_instructions"] = output_parser.get_format_instructions()
            is_sufficient = (
                self.review_prompt.as_runnable(to_dict=True) | agent | itemgetter("output") | output_parser
            ).invoke(inp)["is_sufficient"]
            res = state.copy()
            res["is_sufficient"] = is_sufficient
            return res

        workflow.add_node("search", RunnableLambda(search, name="search"))
        workflow.add_node("review", RunnableLambda(review, name="is_sufficient"))

        workflow.add_edge("search", "review")
        workflow.add_conditional_edges(
            "review", lambda x: x["is_sufficient"] or review_counter >= self.max_tries, {True: END, False: "search"}
        )
        workflow.set_entry_point("search")

        return workflow.compile() | {"collected_context": lambda _: retrieval_helper.viewed_lines}
