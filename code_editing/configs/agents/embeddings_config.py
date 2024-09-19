from dataclasses import dataclass, field

from omegaconf import MISSING


@dataclass
class EmbeddingsConfig:
    _target_: str = MISSING


@dataclass
class OpenAIEmbeddingsConfig(EmbeddingsConfig):
    _target_: str = "langchain_openai.OpenAIEmbeddings"
    model: str = "text-embedding-ada-002"


@dataclass
class HuggingFaceEmbeddingsConfig(EmbeddingsConfig):
    _target_: str = "langchain_community.embeddings.HuggingFaceEmbeddings"
    model_name: str = MISSING
    model_kwargs: dict = field(default_factory=dict)


def setup_embeddings_config(cs):
    cs.store(name="base_openai", group="retrieval/embeddings", node=OpenAIEmbeddingsConfig)
    cs.store(name="base_hf", group="retrieval/embeddings", node=HuggingFaceEmbeddingsConfig)
