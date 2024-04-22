from dataclasses import dataclass


@dataclass
class LoaderConfig:
    _target_: str = "langchain_community.document_loaders.TextLoader"
    autodetect_encoding: bool = True
