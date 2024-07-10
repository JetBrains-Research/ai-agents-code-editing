import os

from langchain_core.documents import Document
from langchain_core.tools import ToolException, tool


def read_file(context, file, start_index):
    with open(file, "r", encoding="utf8", errors="ignore") as f:
        lines = f.readlines()
        line = 0
        char_count = 0
        for i, l in enumerate(lines):
            if char_count + len(l) > start_index:
                line = i
                break
            char_count += len(l)
    line = int(line)
    start = max(0, line - context)
    end = min(len(lines), line + context + 1)
    contents = "".join(lines[start:end])
    return contents, lines, start, end


def read_file_lines(file, line_start, line_end, add_line_numbers=False):
    lines = read_file_full(file).split("\n")
    line_start = max(1, line_start)
    line_end = min(len(lines), line_end)
    res = ""
    for i in range(line_start - 1, line_end):
        if add_line_numbers:
            res += f"{i + 1} {lines[i]}\n"
        else:
            res += f"{lines[i]}\n"
    return res, line_start, line_end, lines


def read_file_full(file) -> str:
    with open(file, "r", encoding="utf8", errors="ignore") as f:
        return f.read()


def write_file_full(file, content):
    with open(file, "w", encoding="utf8", errors="ignore") as f:
        f.write(content)


def my_format_fragment(source: str, start_index: int, page_content: str) -> str:
    res = f"+++ {source}\n"
    res += f"@@ {start_index} @@\n"

    lines = page_content.split("\n")
    res += "\n".join([f" {l}" for l in lines[1:]])

    return res


def my_format_document(doc: Document, _: str) -> str:
    return my_format_fragment(doc.metadata["source"], doc.metadata["start_index"], doc.page_content)


def lines_format_fragment(source: str, start_index: int, page_content: str, repo_path: str) -> str:
    res = f"+++ {source}\n"
    _, _, start_line, _ = read_file(0, parse_file(source, repo_path), start_index)

    lines = page_content.split("\n")[1:]

    for i, l in enumerate(lines):
        res += f" {start_line + i + 1} {l}\n"

    return res


def lines_format_document(doc: Document, repo_path: str) -> str:
    return lines_format_fragment(doc.metadata["source"], doc.metadata["start_index"], doc.page_content, repo_path)


def check_file_inside_repo(file, repo_path):
    # Check if the file is inside the repo, in case .. or / is used to escape the repo
    return os.path.realpath(file).startswith(os.path.realpath(repo_path))


def parse_file(file_name, repo_path):
    """Parse the file and return the full path. Raise ToolException if the file is not valid."""
    file = os.path.join(repo_path, file_name)
    if not os.path.exists(file):
        raise ToolException(f"File {file_name} does not exist")
    if not os.path.isfile(file):
        raise ToolException(f"{file_name} is not a file")
    if not check_file_inside_repo(file, repo_path):
        raise ToolException(f"File {file_name} is not inside the repo")
    return file


@tool
def dummy() -> None:
    """Dummy tool. Does nothing."""
    pass
