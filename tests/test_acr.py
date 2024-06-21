import os.path
from pathlib import Path

from code_editing.agents.context_providers.acr_search import SearchManager


def test_show_definition():
    test_dir = os.path.join(Path(__file__).parents[0], "assets/acr_show_definition")
    search_manager = SearchManager(test_dir, show_lineno=True)

    # Imported function
    res, _, ok = search_manager.show_definition("foo", 3, "b.py")
    assert ok and "1 def foo(a: int, b: int) -> int:" in res
    # Imported class
    res, _, ok = search_manager.show_definition("A", 4, "b.py")
    assert ok and "13 class A:" in res
    # Object of imported class
    res, _, ok = search_manager.show_definition("obj", 5, "b.py")
    assert ok and "13 class A:" in res
    # Method of imported class
    res, _, ok = search_manager.show_definition("foo", 5, "b.py")
    assert ok and "```\n18" in res
    # Imported function
    res, _, ok = search_manager.show_definition("bar", 7, "b.py")
    assert ok and "5 def bar(a: int, b: int) -> int:" in res
    # Reassigned variable
    res, _, ok = search_manager.show_definition("bar", 9, "b.py")
    assert ok and "builtins.int" in res
    # Local function
    res, _, ok = search_manager.show_definition("hello", 13, "b.py")
    assert ok and "13 def hello():" in res
