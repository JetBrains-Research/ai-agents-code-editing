from dataclasses import dataclass


@dataclass
class LoaderConfig:
    target: str = "langchain_community.document_loaders.TextLoader"
    autodetect_encoding: bool = True
