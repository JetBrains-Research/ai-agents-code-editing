import tempfile

from code_editing.agents.graph.collect_edit.editors.util import process_edit


def test_process_edit():
    def mock_edit_func(_: str, snippet: str) -> str:
        return "<edited>" + snippet + "</edited>"

    file = tempfile.mktemp()
    file_content = "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\nline 9\nline 10"
    with open(file, "w") as f:
        f.write(file_content)

    lines_to_edit = [2, 3, 5, 6, 8, 9]
    edited_code = process_edit(file, lines_to_edit, mock_edit_func)

    expected_code = ("line 1\n<edited>line 2\nline 3</edited>\nline 4\n<edited>line 5\nline 6</edited>\nline "
                     "7\n<edited>line 8\nline 9</edited>\nline 10")

    assert edited_code == expected_code
