import re
from typing import Callable, List, Optional, Tuple

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser

from code_editing.agents.tools.common import read_file_full


def process_edit(file, lines_to_edit: List[int], edit_func: Callable[[str, str], str]) -> str:
    """
    Process the edit of a file by editing the lines specified in the list.

    Args:
        file: The file name.
        lines_to_edit: A list of line numbers to edit.
        edit_func: A function that takes the file name and the snippet of code and returns the edited snippet.

    Returns:
        New code for the file.
    """
    # Read the file
    code = read_file_full(file)
    code_lines = code.split("\n")
    # Result code
    result_code = code_lines
    # Split the lines to edit into segments of contiguous lines
    segments = split_into_segments(lines_to_edit)
    # Delta lines (number of lines added or removed)
    delta_lines = -1  # -1 because the first line is 0
    # Process the segments
    for start, end in segments:
        assert start > 0, f"Line numbers should be 1-based, got {start}."
        # Get the snippet of code
        len1 = end - start
        snippet = "\n".join(code_lines[start + delta_lines : end + delta_lines])
        # Edit the snippet
        edited_snippet = edit_func(file, snippet)
        # Replace the snippet in the result code
        len2 = edited_snippet.count("\n") + 1
        result_code = result_code[: start + delta_lines] + edited_snippet.split("\n") + result_code[end + delta_lines :]
        delta_lines += len2 - len1
    # Save the edited code
    return "\n".join(result_code)


def split_into_segments(lines: List[int]) -> List[Tuple[int, int]]:
    """
    Split the list of lines into segments of contiguous lines.

    Args:
        lines: A list of line numbers.

    Returns:
        A list of tuples with the start and end of the segments (exclusive).
    """
    segments = []
    if not lines:
        return segments
    lines = sorted(list(lines))
    start = lines[0]
    for i in range(1, len(lines)):
        if lines[i] != lines[i - 1] + 1:
            segments.append((start, lines[i - 1] + 1))
            start = lines[i]
    segments.append((start, lines[-1] + 1))
    return segments


class MarkdownOutputParser(BaseOutputParser):
    def __init__(self, key: Optional[str] = None):
        super().__init__()
        self.key = key
        if key is not None:
            self.pattern = "```" + key + "\n(.*)\n```"
        else:
            self.pattern = r"```.*?\n(.*)\n```"

    def parse(self, text: str) -> dict:
        matches = re.findall(self.pattern, text, re.DOTALL)
        if not matches:
            raise OutputParserException("No markdown code block found in the text.")
        if len(matches) > 1:
            raise OutputParserException("Multiple markdown code blocks found in the text.")
        return matches[0]

    def get_format_instructions(self) -> str:
        return f'The output should be a markdown code snippet, including the leading "```{self.key or ""}" and trailing "```" tags.'

    @property
    def _type(self) -> str:
        return "markdown_output_parser"

    key: Optional[str] = None
    pattern: str = ""


class TagParser(BaseOutputParser):
    def __init__(self, tag: str):
        super().__init__()
        self.tag = tag

    def parse(self, text: str) -> dict:
        matches = re.findall(f"<{self.tag}>(.*?)</{self.tag}>", text, re.DOTALL)
        return {"matches": matches}

    def get_format_instructions(self) -> str:
        return f'The output should be a text with the tag "<{self.tag}>...</{self.tag}>".'

    @property
    def _type(self) -> str:
        return "tag_parser"

    tag: str = ""
