from abc import abstractmethod
from typing import List

from code_editing.code_editor import CEInput
from code_editing.utils.prompts import prompt_utils
from code_editing.utils.prompts.base_prompt import CEPrompt, ChatMessage


class SimpleCEPrompt(CEPrompt):
    name = "base_simple"

    def __init__(self, max_new_tokens: int):
        super().__init__(max_new_tokens)

    def _base_prompt(self, req: CEInput) -> str:
        message, context = req["instruction"], req["code_base"]
        code = "\n".join([self.file_to_prompt(file_name, file_content) for file_name, file_content in context.items()])
        return prompt_utils.user_prompt_template.format(INSTRUCTION=message, CODE=code)

    @abstractmethod
    def _chat(self, req: CEInput) -> List[ChatMessage]:
        pass

    def chat(self, req: CEInput, has_system_prompt: bool = True) -> List[ChatMessage]:
        res = self._chat(req)
        if not has_system_prompt:
            res = [
                {"role": "user", "content": res[0]["content"]},
                {"role": "assistant", "content": "Understood. Please provide the code base and instruction."},
            ] + res[1:]
        return res

    def file_to_prompt(self, file_name: str, file_content: str) -> str:
        return f"[start of {file_name}]\n{file_content}\n[end of {file_name}]"


class ZeroShotCEPrompt(SimpleCEPrompt):
    name = "zero_shot"

    def __init__(self, max_new_tokens: int):
        super().__init__(max_new_tokens)

    def _chat(self, req: CEInput) -> List[ChatMessage]:
        return [
            {"role": "system", "content": prompt_utils.zero_shot_sys_prompt},
            {"role": "user", "content": self._base_prompt(req)},
        ]


class FewShotCEPrompt(SimpleCEPrompt):
    name = "few_shot"

    example_instruction = prompt_utils.example_instruction
    example_code_base = prompt_utils.example_code_base
    sys_prompt = prompt_utils.sys_prompt
    expected_output = prompt_utils.expected_output

    def __init__(self, max_new_tokens: int):
        super().__init__(max_new_tokens)

    def _chat(self, req: CEInput) -> List[ChatMessage]:
        # Few-shot prompt
        return [
            {
                "role": "system",
                "content": self.sys_prompt,
            },
            {
                "role": "user",
                "content": self._base_prompt(
                    CEInput(instruction=self.example_instruction, code_base=self.example_code_base)
                ),
            },
            {"role": "assistant", "content": self.expected_output},
            {
                "role": "user",
                "content": "Good. Disregard and forget previous gcd code base and instruction. It was for example purposes only. "
                + self._base_prompt(req),
            },
        ]

    def file_to_prompt(self, file_name: str, file_content: str) -> str:
        content_numbered = self.add_line_numbers(file_content)
        return f"[start of {file_name}]\n{content_numbered}\n[end of {file_name}]"

    def add_line_numbers(self, code: str) -> str:
        lines = code.splitlines(keepends=True)
        return "".join(f"<{i + 1}> {line}" for i, line in enumerate(lines))


class FewShotCEPrompt2(FewShotCEPrompt):
    name = "few_shot2"

    sys_prompt = """
You are tasked to generate a diff file based on a given code base and user instructions.
The diff file should represent the necessary changes that, once applied, fulfill the user's request.
Note that the code base may involve multiple files.
Code base includes line numbers for your reference. Remove them when generating the diff file.
Break down the task into steps and explain your planned changes.
Provide the diff in a markdown ```diff``` code block.
""".strip()

    expected_output = """
To achieve this, I will need to:
1. Replace recursion in gcd.py lines 1-4 with a while loop and remove the default value of b
2. In main.py line 6 add argument b=0
Now I will provide the diff:
```diff
--- a/gcd.py
+++ b/gcd.py
@@ -1,4 +1,4 @@
-def gcd(a, b=0):
-    if b == 0:
-        return a
-    return gcd(b, a % b)
+def gcd(a, b):
+    while b:
+        a, b = b, a % b
+    return a
--- a/main.py
+++ b/main.py
@@ -3,5 +3,5 @@ from gcd import gcd

 def main():
     init_db()
-    res = gcd(566)
+    res = gcd(566, 0)
     print(f"GCD is {res}")
```
""".strip()

    def __init__(self, max_new_tokens: int):
        super().__init__(max_new_tokens)
