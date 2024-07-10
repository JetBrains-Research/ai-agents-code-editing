import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


def extract_patch(resp: str) -> Optional[str]:
    resp = str(resp)  # Ensure that the response is a string

    """Extracts code diff from a string containing Markdown code blocks."""
    if resp.strip() == "":
        return ""

    if resp.strip().startswith("diff --git") or resp.strip().startswith("--- a/"):
        # If the response is a diff, return it as is
        return resp

    markdown_matches = re.findall(r"```(?:diff)?\n(.*?)\n```", resp, re.DOTALL)
    if not markdown_matches:
        return None
    res = "\n".join([match for match in markdown_matches])
    return res


def extract_patches(responses: Iterable[str], do_replace_blanks: bool = False) -> List[str]:
    """Extracts code diffs from strings containing Markdown code blocks."""
    res = []
    for resp in responses:
        val = extract_patch(resp)
        res.append(("" if do_replace_blanks else resp) if val is None else val)
    return res


@dataclass
class ParsedDiff:
    """Data class for parsed diff."""

    file_name: str
    old_lines: List[str]
    new_lines: List[str]
    old_line_numbers: List[int]
    new_line_numbers: List[int]


def parse_diff_block(diff_block: str) -> ParsedDiff:
    """
    This function extracts the file name and the changes made in the file from a diff block.

    Args:
        diff_block (str): A string representing a diff block.

    Returns:
        Tuple[str, str]: A tuple containing the file name and the changes made in the file.
    """
    # Extract the file name from the diff block
    file_name = re.search(r"--- a/(.*?)\n", diff_block)
    try:
        file_name = file_name.group(1)
    except AttributeError:
        return ParsedDiff("", [], [], [], [])

    old_lines, new_lines, old_line_numbers, new_line_numbers = [], [], [], []
    segments = diff_block.split("\n@@")[1:]
    for segment in segments:
        # get the line numbers
        line_numbers = re.search(r"^ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", segment)
        start, old_len, new_start, new_len = line_numbers.groups()
        start, old_len, new_start, new_len = int(start), int(old_len or 1), int(new_start), int(new_len or 1)
        # Split the segment into lines
        lines = segment.split("\n")[1:]
        old_block, new_block = "", ""
        for line in lines:
            # If the line starts with "+" or " ", it represents a change
            if line.startswith("+") or line.startswith(" "):
                new_block += line[1:] + "\n"
            if line.startswith("-") or line.startswith(" "):
                old_block += line[1:] + "\n"
        if new_block:
            new_lines.append(new_block)
            new_line_numbers.extend(list(range(new_start, new_start + new_len)))
        if old_block:
            old_lines.append(old_block)
            old_line_numbers.extend(list(range(start, start + old_len)))
    return ParsedDiff(file_name, old_lines, new_lines, old_line_numbers, new_line_numbers)


def parse_diff(diff: str) -> List[ParsedDiff]:
    """
    This function parses a diff and returns a list of ParsedDiff objects.

    Args:
        diff (str): A string representing a diff.

    Returns:
        List[ParsedDiff]: A list of ParsedDiff objects.
    """
    if not diff.strip():
        return []
    # Split the diff into blocks
    blocks = re.split(r"\n(?=diff)", diff)

    parsed_diffs = []
    for block in blocks:
        # Extract the file name and the changes made in the file from the block
        parsed_diff = parse_diff_block(block)
        if parsed_diff.file_name == "":
            continue
        parsed_diffs.append(parsed_diff)
    return parsed_diffs


def create_diff_context(diff: str) -> Dict[str, str]:
    """
    This function creates a context of the changes made in each file from a diff.

    Args:
        diff (str): A string representing a diff.

    Returns:
        Dict[str, str]: A dictionary where the keys are the file names and the values are the changes made in the files.
    """
    parsed_diffs = parse_diff(diff)
    context = {}
    for parsed_diff in parsed_diffs:
        file_name, file_content = parsed_diff.file_name, "# ...\n".join(parsed_diff.new_lines)
        context[file_name] = file_content
    return context


def compare_diffs(diff1: str, diff2: str) -> Tuple[List[str], List[str]]:
    """
    This function compares two diffs and returns the changes made in each file for both diffs.

    Args:
        diff1 (str): A string representing the first diff.
        diff2 (str): A string representing the second diff.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists. The first list contains the changes made in each file for the first diff. The second list contains the changes made in each file for the second diff.
    """
    # Create a context of the changes made in each file for both diffs
    context1 = create_diff_context(diff1)
    context2 = create_diff_context(diff2)

    # Get the union of the keys from both contexts
    keys = set(context1.keys()).union(context2.keys())

    changes1, changes2 = [], []
    for key in keys:
        # Get the changes made in the file for both diffs. If no changes were made in a file, add "<NO CHANGES>"
        changes1.append(context1.get(key, "<NO CHANGES>"))
        changes2.append(context2.get(key, "<NO CHANGES>"))
    return changes1, changes2


def edited_lines_per_file(diff: str) -> Dict[str, List[int]]:
    """
    This function returns the line numbers that were edited in each file from a diff.

    Args:
        diff (str): A string representing a diff.

    Returns:
        Dict[str, List[int]]: A dictionary where the keys are the file names and the values are the numbers of the lines that were edited in the files.
    """
    parsed_diffs = parse_diff(diff)
    edited_lines = {}
    for parsed_diff in parsed_diffs:
        edited_lines[parsed_diff.file_name] = parsed_diff.old_line_numbers
    return edited_lines
