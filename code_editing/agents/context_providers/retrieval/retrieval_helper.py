import logging
import os.path
from abc import abstractmethod
from typing import List

from hydra.utils import get_class
from langchain.text_splitter import TextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, format_document

from code_editing.agents.context_providers.context_provider import ContextProvider
from code_editing.agents.context_providers.retrieval.file_extensions import extensions, filter_docs
from code_editing.agents.tools.common import parse_file, read_file
from code_editing.configs.agents.context_providers.loader_config import LoaderConfig
from code_editing.utils.git_utils import get_head_sha_unsafe


class RetrievalHelper(ContextProvider):
    def __init__(
        self,
        repo_path: str,
        data_path: str,
        splitter: TextSplitter,
        loader: LoaderConfig
    ):
        """
        RetrievalHelper

        Helper class for retrieving documents using Faiss vector store.
        Handles loading, embedding, indexing, db saving and loading.
        """
        self.repo_path = repo_path
        self.data_path = data_path

        # Create a vector store directory
        self.vector_path = os.path.join(data_path, "vector_store")
        os.makedirs(self.vector_path, exist_ok=True)

        # Set log level for the directory loader
        logging.getLogger("langchain_community.document_loaders.directory").setLevel(logging.ERROR)
        loader_cls = get_class(loader["target"])
        loader_kwargs = dict(loader)
        loader_kwargs.pop("target")
        if "autodetect_encoding" not in loader_kwargs:
            loader_kwargs["autodetect_encoding"] = True
        self.loader_kwargs = loader_kwargs
        self.loader_cls = loader_cls

        self.splitter = splitter
        self.doc_prompt = PromptTemplate.from_template("# Path: {source}\n{page_content}")

        # Initialize the record manager and the db
        self.namespace = (
            os.path.basename(os.path.normpath(self.repo_path))
            + "__"
            + get_head_sha_unsafe(self.repo_path, self.data_path)
        )
        self.global_record_manager_path = os.path.join(self.vector_path, f"{self.namespace}.sqlite")

        self._init_db()

        # Lines viewed storage
        self.viewed_lines = {}
        self.logger = logging.getLogger("agents.retrieval_helper")

    @abstractmethod
    def _init_db(self):
        pass

    @abstractmethod
    def search(self, query: str, k: int, run_manager=None, callbacks=None) -> List[Document]:
        pass

    @abstractmethod
    def reindex_incremental(self, docs: List[Document]):
        pass

    def reindex_files(self, files: List[str]):
        """Reindex the documents in the repo that are in the file list."""
        docs = []
        for file in files:
            loader = self.loader_cls(file_path=file, **self.loader_kwargs)
            docs += loader.load()
        docs = self._prep_documents(docs)
        # Only reindex the documents that are in the file list
        self.reindex_incremental(docs)

    def add_changed_file(self, file: str):
        """Update the vector store with the changed file. Also save the changed file for reset."""
        self.reindex_files([file])

    def add_viewed_docs(self, docs: List[Document]):
        """Save the viewed lines for the localization evaluation."""
        for doc in docs:
            file_name = doc.metadata["source"]
            start_index = doc.metadata["start_index"]
            file = parse_file(file_name, self.repo_path)
            _, _, start, _ = read_file(0, file, start_index)
            end = start + doc.page_content.count("\n") + 1
            self.add_viewed_doc(file_name, start + 1, end + 1)

    def add_viewed_doc(self, file_name, start_line, end_line):
        self.viewed_lines.setdefault(file_name, set()).update(range(start_line, end_line))

    def _prep_documents(self, docs: List[Document]) -> List[Document]:
        docs = list(filter_docs(docs))
        docs = list(self.splitter.transform_documents(docs))
        docs = self._add_doc_headers(docs)
        return docs

    def _get_all_documents(self) -> List[Document]:
        docs = []
        for ext in extensions:
            glob = "**/*." + ext
            dir_loader = DirectoryLoader(
                self.repo_path,
                loader_cls=self.loader_cls,
                loader_kwargs=self.loader_kwargs,
                silent_errors=True,
                glob=glob,
                recursive=True,
            )
            docs += dir_loader.load()
        docs = self._prep_documents(docs)
        return docs

    def _add_doc_headers(self, docs: List[Document]) -> List[Document]:
        for doc in docs:
            rel_src = os.path.relpath(doc.metadata["source"], self.repo_path)
            doc.metadata["source"] = rel_src.replace("\\", "/")
            doc.page_content = format_document(doc, self.doc_prompt)
        return docs

    def _placeholder_docs(self):
        """Return placeholder documents to initialize the vector store."""
        return [
            Document(
                "placeholder",
                metadata={
                    "source": "/placeholder",
                    "start_index": 0,
                },
            )
        ]

    loader_cls = None
    loader_kwargs = None
