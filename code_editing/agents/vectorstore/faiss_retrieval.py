import logging
import os
import shutil
import tempfile
from typing import List

from langchain.indexes import SQLRecordManager, index
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import create_engine

from code_editing.agents.vectorstore.retriever_helper import RetrievalHelper
from code_editing.utils.wandb_utils import get_current_ms


class FaissRetrieval(RetrievalHelper):
    def __init__(self, embeddings: OpenAIEmbeddings, **kwargs):
        self.embeddings = embeddings
        super().__init__(**kwargs)

    db: FAISS = None

    def search(self, query: str, k: int, run_manager=None, callbacks=None) -> List[Document]:
        if run_manager is not None:
            callbacks = run_manager.get_child()
        return self.db.as_retriever(k=k).get_relevant_documents(
            query, callbacks=callbacks
        )

    def _init_db(self):
        logging.getLogger("faiss.loader").setLevel(logging.WARNING)
        # Invariant: the saved db contents correspond to the unchanged state of the repo at base commit
        is_first_time = not os.path.exists(os.path.join(self.vector_path, f"{self.namespace}.faiss"))
        if is_first_time:
            # Create a new vector store with placeholder documents
            self.logger.info(f"No vector store found for {self.namespace}. Creating a new one. This may take a while.")
            start_ms = get_current_ms()
            self.db = FAISS.from_documents(self._placeholder_docs(), self.embeddings)
            self.db.as_retriever()
            # Connect to the global record manager and initialize a corresponding namespace
            db_url = "sqlite:///" + self.global_record_manager_path
            global_record_manager = SQLRecordManager(self.namespace, db_url=db_url)
            global_record_manager.create_schema()
            # Index all the documents, save the record manager for the future use
            self._reindex_full(global_record_manager)
            # Save the db to the disk
            self.db.save_local(self.vector_path, index_name=self.namespace)
            self.logger.info(
                f"Vector store created for {self.namespace} in {round((get_current_ms() - start_ms) / 1000, 2)} seconds."
            )
        else:
            # Load the existing vector store
            self.db = FAISS.load_local(self.vector_path, self.embeddings, index_name=self.namespace)
        self._init_record_manager()

    def __del__(self):
        if hasattr(self, "record_manager"):
            # Close the db connection
            self._engine.dispose(close=True)
            # Delete the temp file for the local record manager
            os.remove(self.record_manager_path)

    def _init_record_manager(self):
        """Initialize a record manager for the current state of the vector store.

        After the run is finished, the record manager should be deleted.

        The record manager is initialized from the global record manager.
        """
        # Initialize the record manager for tracking the current state of the vector store
        self.record_manager_path = tempfile.mktemp(suffix=".sqlite")
        # Check that the global record manager exists
        if not os.path.exists(self.global_record_manager_path):
            raise ValueError(f"Prepared record manager not found at {self.global_record_manager_path}")
        # Copy the global record manager to the local record manager
        shutil.copyfile(self.global_record_manager_path, self.record_manager_path)
        # Connect to the record manager
        db_url = "sqlite:///" + self.record_manager_path
        self._engine = create_engine(db_url)  # HACK: our engine to disable the process blocking the file
        self.record_manager = SQLRecordManager(self.namespace, engine=self._engine)

    def _reindex_full(self, record_manager):
        """Reindex all the documents in the repo. This is a long operation."""
        docs = self._get_all_documents()
        if len(docs) > 10000:
            self.logger.warning(f"Found {len(docs)} documents in {self.namespace}. This may take a while to process.")
        index(docs, record_manager, self.db, cleanup="full", source_id_key="source")

    def reindex_incremental(self, docs: List[Document]):
        index(docs, self.record_manager, self.db, cleanup="incremental", source_id_key="source")
