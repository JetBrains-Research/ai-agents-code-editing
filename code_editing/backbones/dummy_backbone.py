import random

from code_editing.backbones.base_backbone import CEBackbone, CEInput, CEOutput
from code_editing.prompts import CEPrompt
from code_editing.utils.tokenization_utils import TokenizationUtils


class DummyBackbone(CEBackbone):
    """This is a dummy backbone that generates a diff randomly from the code base.

    It picks a random file, and then picks random lines from that file, and then generates a diff from those lines.

    This backbone is used for testing purposes and as a baseline for other backbones.
    """
    name = "dummy"

    def __init__(self, model_name: str, prompt: CEPrompt, file_sample_p: float, line_sample_p: float):
        self.gen_length = prompt.max_new_tokens
        self._tok = TokenizationUtils(model_name)
        self.prompt = prompt
        self.file_sample_p = file_sample_p
        self.line_sample_p = line_sample_p

    def generate_diff(self, req: CEInput, **kwargs) -> CEOutput:
        diffs = []
        for file_name, file_contents in req["code_base"].items():
            if random.random() > self.file_sample_p:
                continue
            diffs.append((file_name, self._expand_to_diff(file_contents)))
        output = '```diff\n'
        output += '\n'.join([f'--- a/{file_name}\n+++ b/{file_name}\n{self._truncate_diff(diff, len(diffs))}' for file_name, diff in diffs])
        output += '\n```'
        return self.prompt.postprocess(req, {"prediction": output})

    def _expand_to_diff(self, code):
        res = ''
        for line in code.split('\n'):
            if random.random() > self.line_sample_p:
                continue
            res += f'+{line}\n'
            res += f'-{line}\n'
        return res

    def _truncate_diff(self, diff: str, cnt: int = 1) -> str:
        return self._tok.truncate_text(diff, self.gen_length // cnt)
