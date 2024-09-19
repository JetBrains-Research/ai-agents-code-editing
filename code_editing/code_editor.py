from abc import ABC, abstractmethod
from typing import Dict, List, TypedDict

from typing_extensions import NotRequired  # we use python 3.10


class CEInput(TypedDict):
    """This is a TypedDict class that represents the input to the code editor"""

    instruction: str
    code_base: Dict[str, str]
    instance_id: NotRequired[str]
    raw_data: NotRequired[Dict]


class BackBoneOutput(TypedDict):
    """This is a TypedDict class that represents the output of the backbone"""

    """Prediction output of the code editing model."""
    prediction: str


class CEOutput(BackBoneOutput, total=False):
    """This is a TypedDict class that represents the output of the code editor"""

    """Mapping from file names to the line numbers that were viewed by the model. This is used for the localization evaluation."""
    viewed_lines: Dict[str, List[int]]


class CEBackbone(ABC):
    """
    This is an abstract class that represents the code editing backbone.

    Backbones are responsible for generating the code editing diff given the input.
    """

    name: str = "base"

    @abstractmethod
    def generate_diff(self, req: CEInput, **kwargs) -> CEOutput:
        """
        This method generates the code editing diff given the input.

        @param req: The input to the code editing model.
        @param kwargs: Additional keyword arguments.
        @return: The output of the code editing model.
        """
        pass


class CodeEditor(ABC):
    @abstractmethod
    def generate_diff(self, req: CEInput) -> CEOutput:
        pass

    @property
    def metadata(self) -> dict:
        return {"type": "base"}
