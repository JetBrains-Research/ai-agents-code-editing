from code_editing.code_editor import CEBackbone, CEInput, CEOutput, CodeEditor
from code_editing.utils import wandb_utils
from code_editing.utils.preprocessors.base_preprocessor import CEPreprocessor


class CEBaseline(CodeEditor):
    def __init__(self, backbone: CEBackbone, preprocessor: CEPreprocessor):
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.run_name = backbone.name

    def generate_diff(self, req: CEInput) -> CEOutput:
        # Preprocess the input
        start_ms = wandb_utils.get_current_ms()
        old_req = req
        req = self.preprocessor(req)
        after_preprocess_ms = wandb_utils.get_current_ms()
        # Generate the diff using the backbone
        resp = self.backbone.generate_diff(req)

        return resp

    @property
    def metadata(self) -> dict:
        return {
            "type": "baseline",
            "backbone_name": self.backbone.name,
            "preprocessor_name": self.preprocessor.name,
        }
