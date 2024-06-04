from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from code_editing.agents.context_providers.retrieval.retrieval_helper import RetrievalHelper


class BM25Retrieval(RetrievalHelper):
    def search(self, query: str, k: int, run_manager=None, callbacks=None) -> List[Document]:
        if run_manager is not None:
            callbacks = run_manager.get_child()

        docs = self._get_all_documents()
        bm25 = BM25Retriever.from_documents(docs, k=k)

        return bm25.get_relevant_documents(query, callbacks=callbacks)

    def _init_db(self):
        pass

    def reindex_incremental(self, docs: List[Document]):
        pass
