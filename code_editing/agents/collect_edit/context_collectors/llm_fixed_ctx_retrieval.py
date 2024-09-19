from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from code_editing.agents.collect_edit.context_collectors.llm_retrieval import LLMRetrieval
from code_editing.agents.run import AgentRunManager
from code_editing.agents.tools.common import parse_file, read_file_full
from code_editing.utils.tokenization_utils import TokenizationUtils


class LLMFixedCtxRetrieval(LLMRetrieval):
    name = "llm_fixed_ctx_retrieval"

    def __init__(self, total_context: int = 10000, max_searches=10, **kwargs):
        super().__init__(**kwargs)
        self.total_context = total_context
        self.max_searches = max_searches
        self.tok_utils = TokenizationUtils("gpt-3.5-turbo-16k")

    @property
    def _runnable(self):
        # FIXME
        raise NotImplementedError("This agent was broken by the langgraph refactor")

        retrieval_helper = self.get_ctx_provider("retrieval_helper")
        agent_executor = self.react_agent(
            tools=self.get_llm_retrieval_tools(retrieval_helper),
            # memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="output"),
        )

        workflow = StateGraph(dict)

        searches_counter = 0

        def search(state):
            nonlocal searches_counter
            searches_counter += 1

            inp = state.copy()
            if "context_size" not in inp:
                inp["context_size"] = 0
                inp["intermediate_steps"] = ""
            inp["total_context"] = self.total_context
            out = (self.search_prompt.as_runnable(to_dict=True) | agent_executor).invoke(inp)
            res = state.copy()
            res["intermediate_steps"] = str(retrieval_helper.viewed_lines)
            # calc context size
            context_size = 0
            for file_name, lines in retrieval_helper.viewed_lines.items():
                file = parse_file(file_name, retrieval_helper.repo_path)
                code_lines = read_file_full(file).split("\n")
                for line in lines:
                    context_size += self.tok_utils._count_tokens_completion(code_lines[line - 1])
            res["context_size"] = context_size
            return res

        def do_continue(state):
            return state.get("context_size", 0) < self.total_context and searches_counter < self.max_searches

        workflow.add_node("search", RunnableLambda(search, name="search"))
        workflow.add_conditional_edges("search", do_continue, {True: "search", False: END})
        workflow.set_entry_point("search")

        return workflow.compile() | {"collected_context": lambda _: retrieval_helper.viewed_lines}
