from abc import ABC, abstractmethod
from typing import Iterable, Iterator, Tuple, Sized

from code_editing.backbones.base_backbone import CEInput


class CEDataSource(ABC, Sized, Iterable[Tuple[CEInput, dict]]):
    """Code editing data source. Iterates over (input, metadata) pairs."""
    name = "base"

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[CEInput, dict]]:
        pass
