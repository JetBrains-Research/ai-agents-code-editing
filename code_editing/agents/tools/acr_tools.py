from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, Field

from code_editing.agents.context_providers.acr_search import SearchManager
from code_editing.agents.tools.base_tool import CEBaseTool


def class_for_acr(_name: str, _description: str, _args_schema: Type[BaseModel], run_tool: Callable) -> Type[CEBaseTool]:
    class ACRTool(CEBaseTool):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.args_schema = _args_schema
            if self.dry_run:
                return
            self.search_manager = self.get_ctx_provider(SearchManager)

        name = _name
        description = _description
        args_schema = _args_schema

        def _run_tool(self, *args, **kwargs) -> str:
            return run_tool(self.search_manager, *args, **kwargs)

        @property
        def short_name(self) -> Optional[str]:
            return "acr"

        search_manager: SearchManager = None

    return ACRTool


class ACRSearchClassInput(BaseModel):
    class_name: str = Field(description="Name of the class to search for", examples=["MyClass"])


ACRSearchClass = class_for_acr(
    "search-class",
    "Search for a class in the codebase.",
    ACRSearchClassInput,
    lambda search_manager, class_name: search_manager.search_class(class_name)[0],
)


class ACRSearchMethodInFileInput(BaseModel):
    method_name: str = Field(description="Name of the method to search for", examples=["get_user_by_id"])
    file_path: str = Field(description="Path to the file to search in", examples=["path/to/file.py"])


ACRSearchMethodInFile = class_for_acr(
    "search-method-in-file",
    "Search for a method in a given file.",
    ACRSearchMethodInFileInput,
    lambda search_manager, method_name, file_path: search_manager.search_method_in_file(method_name, file_path)[0],
)


class ACRSearchMethodInClassInput(BaseModel):
    method_name: str = Field(description="Name of the method to search for", examples=["get_user_by_id"])
    class_name: str = Field(description="Name of the class to search in", examples=["MyClass"])


ACRSearchMethodInClass = class_for_acr(
    "search-method-in-class",
    "Search for a method in a given class.",
    ACRSearchMethodInClassInput,
    lambda search_manager, method_name, class_name: search_manager.search_method_in_class(method_name, class_name)[0],
)


class ACRSearchMethodInput(BaseModel):
    method_name: str = Field(description="Name of the method to search for", examples=["get_user_by_id"])


ACRSearchMethod = class_for_acr(
    "search-method",
    "Search for a method in the entire codebase.",
    ACRSearchMethodInput,
    lambda search_manager, method_name: search_manager.search_method(method_name)[0],
)


class ACRSearchCodeInput(BaseModel):
    code_str: str = Field(description="Code snippet to search for", examples=["import os"])


ACRSearchCode = class_for_acr(
    "search-code",
    "Search for a code snippet in the entire codebase.",
    ACRSearchCodeInput,
    lambda search_manager, code_str: search_manager.search_code(code_str)[0],
)


class ACRSearchCodeInFileInput(BaseModel):
    code_str: str = Field(description="Code snippet to search for", examples=["import os"])
    file_path: str = Field(description="Path to the file to search in", examples=["path/to/file.py"])


ACRSearchCodeInFile = class_for_acr(
    "search-code-in-file",
    "Search for a code snippet in a given file.",
    ACRSearchCodeInFileInput,
    lambda search_manager, code_str, file_path: search_manager.search_code_in_file(code_str, file_path)[0],
)


class ACRShowDefinitionInput(BaseModel):
    symbol: str = Field(description="Code symbol to show definition for", examples=["model", "foo", "A"])
    line_number: int = Field(description="Line number (1-indexed) containing the symbol", examples=[1, 10])
    file_path: str = Field(description="Path with the symbol", examples=["path/to/file.py"])


ACRShowDefinition = class_for_acr(
    "show-definition",
    "Show the code where this symbol (variable/method/function/class) is defined.",
    ACRShowDefinitionInput,
    lambda search_manager, symbol, line_number, file_path: search_manager.show_definition(
        symbol, int(line_number), file_path
    ),
)
