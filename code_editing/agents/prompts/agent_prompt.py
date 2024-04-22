import re
from typing import List

from langchain_core.tools import ToolException

from code_editing.backbones.base_backbone import CEInput, CEOutput
from code_editing.prompts.base_prompt import ChatMessage
from code_editing.prompts.ce_prompts import SimpleCEPrompt


class SimpleAgentPrompt(SimpleCEPrompt):
    name = "agent_prompt"

    def __init__(self, max_new_tokens: int):
        super().__init__(max_new_tokens)

    def _chat(self, req: CEInput) -> List[ChatMessage]:
        code = '\n'.join(
            [self.file_to_prompt(file_name, file_content) for file_name, file_content in req["code_base"].items()])
        return [
            {"role": "system",
             "content": "You will edit the code fragment to follow the provided instruction. Please print out the "
                        "edited code fragment in the same format. Don't change the file names or the line numbers. "
                        "Please use the same format [start of filename] ... [end of filename] to mark the edited code."},
            {"role": "user",
             "content": f"INSTRUCTION: {req['instruction']}\nCODE:\n{code}\n DONE: Now please provide the edited code "
                        f"fragments in the same format."}
        ]

    def postprocess(self, req: CEInput, resp: CEOutput) -> CEOutput:
        matches = re.findall(r"\[start of (.*?)]\n(.*?)\n\[end of \1]", resp['prediction'], flags=re.DOTALL)
        if not matches:
            raise ToolException("Edit failed. Code editing response is not in the expected format. Please try again.")
        result = resp.copy()
        result['prediction'] = matches[0][1]
        return result
