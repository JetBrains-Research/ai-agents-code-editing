from code_editing.backbones.base_backbone import CEInput
from code_editing.preprocessors.base_preprocessor import CEPreprocessor
from code_editing.utils.tokenization_utils import TokenizationUtils


class TruncationCEPreprocessor(CEPreprocessor):
    name = "truncate"

    def __init__(self, model_name: str, max_length: int):
        self.tok_utils = TokenizationUtils(model_name)
        self.max_length = max_length

    def __call__(self, req: CEInput) -> CEInput:
        files = len(req['code_base'])
        res = req.copy()
        for file_name, file_content in req['code_base'].items():
            res['code_base'][file_name] = self.tok_utils.truncate_text(file_content, self.max_length // files)
        return res
