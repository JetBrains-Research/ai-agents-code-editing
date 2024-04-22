from abc import ABC, abstractmethod

from code_editing.backbones.base_backbone import CEInput


class CEPreprocessor(ABC):
    name = "base"

    @abstractmethod
    def __call__(self, req: CEInput) -> CEInput:
        pass
