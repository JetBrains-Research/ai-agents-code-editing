from abc import ABC, ABCMeta, abstractmethod


class ContextProvider(ABC):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, repo_path: str, data_path: str, *args, **kwargs):
        pass
