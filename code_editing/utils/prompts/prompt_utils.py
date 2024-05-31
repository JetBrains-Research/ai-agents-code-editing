sys_prompt = """
You are tasked to generate a diff file based on a given code base and user instructions.
The diff file should represent the necessary changes that, once applied, fulfill the user's request.
Note that the code base may involve multiple files.
Code base includes line numbers for your reference. Remove them when generating the diff file.
Explain your planned changes and add diff code blocks for your proposed modifications.
""".strip()

example_instruction = (
    """Rewrite the function gcd(a, b) to use a while loop instead of recursion. Remove the default value of b."""
)

example_code_base = {
    "main.py": """
from utils import init_db
from gcd import gcd

def main():
    init_db()
    res = gcd(566)
    print(f"GCD is {res}")
""".strip(),
    "gcd.py": """
def gcd(a, b=0):
    if b == 0:
        return a
    return gcd(b, a % b)
""".strip(),
}

expected_output = """
Here are the changes that need to be made:
1. Replace recursion in gcd.py lines 1-4 with a while loop and remove the default value of b in line 1
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
```
2. In main.py line 6 add argument b=0
```diff
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

user_prompt_template = """
Generate a diff for the following instruction and code base:
<instruction>
{INSTRUCTION}
</instruction>
<code>
{CODE}
</code>
""".strip()

zero_shot_sys_prompt = """
You will be provided with a partial code base and an instruction to modify it by the user. You will need to generate a patch file that can be applied to the code base to perform the instruction.
Here is an example of an instruction:
<instruction>
Rewrite the function euclidean to use recursion instead of a while loop.
</instruction>
Here is an example of a partial code base.
<code>
[start of file.py]
def euclidean(a, b):
    while b:
    a, b = b, a % b
    return a
[end of file.py]
</code>
Here is an example of a patch file for this request. It consists of changes to the code base. It specifies the file names, the line numbers of each change, and the removed and added lines. A single patch file can contain changes to multiple files.
```diff
--- a/file.py
+++ b/file.py
@@ -1,27 +1,35 @@
def euclidean(a, b):
- while b:
- a, b = b, a % b
- return a
+ if b == 0:
+ return a
+ return euclidean(b, a % b)
```
I need you to read the user's instruction and code base, and generate a patch file that can be applied to the code base to perform the instruction. Don't output anything besides the patch file.
""".strip()
