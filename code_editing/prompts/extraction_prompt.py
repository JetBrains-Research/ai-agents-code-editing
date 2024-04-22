import difflib
import re
from collections import defaultdict
from typing import List

from code_editing.backbones.base_backbone import CEInput, CEOutput
from code_editing.prompts.base_prompt import ChatMessage
from code_editing.prompts.ce_prompts import SimpleCEPrompt


class CodeFragmentCEPrompt(SimpleCEPrompt):
    name = "extract_fragment"

    def __init__(self, max_new_tokens: int):
        super().__init__(max_new_tokens)

    def _chat(self, req: CEInput) -> List[ChatMessage]:
        code = '\n'.join(
            [self.file_to_prompt(file_name, file_content) for file_name, file_content in req["code_base"].items()])
        return [
            {"role": "system",
             "content": "You will edit the code fragments to follow the provided instruction. Please print out the "
                        "edited code fragments in the same format. Don't change the file names or the line numbers. "
                        "Break down the task into smaller steps first and then edit the code fragments accordingly. "},
            {"role": "user",
             "content": f"INSTRUCTION: {req['instruction']}\nCODE: {code}\n DONE: Now please provide the edited code "
                        f"fragments in the same format."}
        ]

    def postprocess(self, req: CEInput, resp: CEOutput) -> CEOutput:
        return to_diff(req, resp)


def to_diff(req: CEInput, resp: CEOutput) -> CEOutput:
    # TODO: write tests for this function (very complex)
    # Convert many updated files to a diff

    old_code_base = req['code_base'].copy()

    mods = re.findall(r'\[start of (.*?)]\n(.*?)\n\[end of \1]', resp['prediction'], flags=re.DOTALL)
    # Group the modifications by file
    files = defaultdict(list)
    for file_name_line, file_content in mods:
        file_name, lines = file_name_line.split('#')
        files[file_name].append((lines, file_content))

    # Build the diff file by file
    final_diff = ""
    for file_name, lines_and_files in files.items():
        file_diff = []
        # Sort code fragments by line number
        lines_and_files.sort(key=lambda x: int(x[0][1:]))
        # Delta lines is used to keep track of the difference in line numbers between the old and new code
        delta_lines = 0
        # Iterate over the code fragments
        for lines, new_code in lines_and_files:
            # Get the old and new code
            old_lines = old_code_base.get(f"{file_name}#{lines}", None)
            if old_lines is None:
                # The code fragment was not in the original code, LLM broke it
                continue
            old_lines = old_lines.split('\n')
            new_lines = new_code.split('\n')

            # Get the start line of the code fragment in the reference code
            ref_start = int(lines[1:])

            # Get the diff, without any other context
            diff_lines = difflib.unified_diff(old_lines, new_lines, n=3)
            diff_lines = list(diff_lines)
            if len(diff_lines) <= 3:
                # No changes
                continue
            block_diff = '\n'.join(diff_lines[2:])

            # Modifying the headers
            def replace_line_numbers(match: re.Match[str]):
                nonlocal delta_lines

                old_start_fragment, old_length, new_start_fragment, new_length = match.groups()
                old_start_fragment = int(old_start_fragment)
                old_length = int(old_length) if old_length else 1
                new_start_fragment = int(new_start_fragment)
                new_length = int(new_length) if new_length else 1
                # Update the header
                old_start_fragment += ref_start - 1
                new_start_fragment += ref_start - 1 + delta_lines
                delta_lines += new_length - old_length
                return f'@@ -{old_start_fragment},{old_length} +{new_start_fragment},{new_length} @@'

            modified_block_diff = re.sub(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@\n', replace_line_numbers,
                                         block_diff, flags=re.DOTALL)
            file_diff.append(modified_block_diff)
        file_diff = '\n'.join(file_diff)
        if file_diff:
            # If there are changes, add the file diff to the final diff
            final_diff += f"diff --git a/{file_name} b/{file_name}\n"
            final_diff += f"--- a/{file_name}\n+++ b/{file_name}\n"
            final_diff += f"{file_diff}\n"
    resp['prediction'] = f"```diff\n{final_diff}\n```"
    return resp
