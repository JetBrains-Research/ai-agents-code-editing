from typing import Optional

from langchain_core.runnables import RunnableLambda

from code_editing.agents.agent_graph import AgentGraph
from code_editing.agents.collect_edit.collect_edit import CollectEditState
from code_editing.agents.context_providers.retrieval.retrieval_helper import RetrievalHelper
from code_editing.utils.tokenization_utils import TokenizationUtils


class AsIsRetrieval(AgentGraph):
    name = "as_is_retrieval"

    def __init__(self, k: Optional[int] = 10, total_context: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.total_context = total_context

        if (k is None) == (total_context is None):
            raise ValueError("Either k or total_context should be provided")

    @property
    def _runnable(self):
        retrieval_helper = self.get_ctx_provider(RetrievalHelper)

        def search(state: CollectEditState, config) -> CollectEditState:
            if self.k is not None:
                docs = retrieval_helper.search(state["instruction"], k=self.k, callbacks=config["callbacks"])
            else:
                tok_utils = TokenizationUtils("gpt-3.5-turbo-16k")

                raw_docs = retrieval_helper.search(state["instruction"], 100, callbacks=config["callbacks"])
                docs = []
                buf_len = 0
                max_len = self.total_context
                for doc in raw_docs:
                    if buf_len > max_len - 10:
                        break
                    truncated = tok_utils.truncate_text(doc.page_content, max_len - buf_len)
                    # noinspection PyProtectedMember
                    buf_len += tok_utils._count_tokens_completion(truncated)
                    docs.append(doc)
            # Save viewed docs
            retrieval_helper.add_viewed_docs(docs)

            res = state.copy()
            res["collected_context"] = retrieval_helper.viewed_lines
            return res

        return RunnableLambda(search, name="As Is Retrieval")
