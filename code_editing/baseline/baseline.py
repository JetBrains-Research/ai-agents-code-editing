from abc import ABC, abstractmethod

import wandb
from wandb.sdk.data_types.trace_tree import StatusCode

from code_editing.backbones import CEBackbone
from code_editing.backbones.base_backbone import CEInput, CEOutput
from code_editing.preprocessors.base_preprocessor import CEPreprocessor
from code_editing.utils import wandb_utils


class CodeEditor(ABC):
    run_name = "base"

    @abstractmethod
    def generate_diff(self, req: CEInput) -> CEOutput:
        pass

    @property
    def metadata(self) -> dict:
        return {"type": "base"}


class CEBaseline(CodeEditor):
    def __init__(self, backbone: CEBackbone, preprocessor: CEPreprocessor):
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.run_name = backbone.name

    def generate_diff(self, req: CEInput) -> CEOutput:
        # Initialize the root span for W&B
        root_span = None
        if wandb.run is not None:
            root_span = wandb_utils.build_main_trace(
                req,
                wandb_utils.get_current_ms(),
                "Code Editing",
                metadata={
                    "preprocessor_name": self.preprocessor.name,
                    "backbone_name": self.backbone.name,
                },
            )

        # Preprocess the input
        start_ms = wandb_utils.get_current_ms()
        old_req = req
        req = self.preprocessor(req)
        after_preprocess_ms = wandb_utils.get_current_ms()
        # Log the preprocessing trace to W&B
        if wandb.run is not None:
            wandb_utils.log_preprocessor_trace(old_req, req, start_ms, after_preprocess_ms, root_span)

        # Generate the diff using the backbone
        try:
            resp = self.backbone.generate_diff(req, parent_span=root_span)
            if wandb.run is not None:
                wandb_utils.log_main_trace(root_span, old_req, resp, StatusCode.SUCCESS)
        except Exception as e:
            if wandb.run is not None:
                wandb_utils.log_main_trace(root_span, old_req, None, StatusCode.ERROR, str(e))
            raise e

        return resp

    @property
    def metadata(self) -> dict:
        return {
            "type": "baseline",
            "backbone_name": self.backbone.name,
            "preprocessor_name": self.preprocessor.name,
        }
