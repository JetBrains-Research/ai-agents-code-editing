from abc import ABC, abstractmethod

from code_editing.code_editor import CEInput


class CEPreprocessor(ABC):
    name = "base"

    @abstractmethod
    def __call__(self, req: CEInput) -> CEInput:
        pass
