from typing import Optional, Any

from pydantic import BaseModel, Field

from code_editing.agents.tools.base_tool import CEBaseTool

"""
search_class(class_name: str): Search for a class in the codebase"
search_method_in_file(method_name: str, file_path: str): Search for a method in a given file"
search_method_in_class(method_name: str, class_name: str): Search for a method in a given class"
search_method(method_name: str): Search for a method in the entire codebase"
search_code(code_str: str): Search for a code snippet in the entire codebase"
search_code_in_file(code_str: str, file_path: str): Search for a code snippet in a given file file"
"""


class ACRSearchClass(CEBaseTool):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args_schema = self.ACRSearchClassInput
        if self.dry_run:
            return
        if "search_manager" not in kwargs:
            raise ValueError("Search manager is required for code search tool")
        self.search_manager = kwargs.get("search_manager")

    class ACRSearchClassInput(BaseModel):
        class_name: str = Field(description="Name of the class to search for", examples=["MyClass"])

    name = "search-class"
    description = """
    Search for a class in the codebase.
    """
    args_schema = ACRSearchClassInput

    def _run(self, class_name: str) -> str:
        return self.search_manager.search_class(class_name)[0]

    @property
    def short_name(self) -> Optional[str]:
        return "acr"

    search_manager: Any = None


class ACRSearchMethodInFile(CEBaseTool):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args_schema = self.ACRSearchMethodInFileInput
        if self.dry_run:
            return
        if "search_manager" not in kwargs:
            raise ValueError("Search manager is required for code search tool")
        self.search_manager = kwargs.get("search_manager")

    class ACRSearchMethodInFileInput(BaseModel):
        method_name: str = Field(description="Name of the method to search for", examples=["get_user_by_id"])
        file_path: str = Field(description="Path to the file to search in", examples=["path/to/file.py"])

    name = "search-method-in-file"
    description = """
    Search for a method in a given file.
    """
    args_schema = ACRSearchMethodInFileInput

    def _run(self, method_name: str, file_path: str) -> str:
        return self.search_manager.search_method_in_file(method_name, file_path)[0]

    @property
    def short_name(self) -> Optional[str]:
        return "acr"

    search_manager: Any = None


class ACRSearchMethodInClass(CEBaseTool):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args_schema = self.ACRSearchMethodInClassInput
        if self.dry_run:
            return
        if "search_manager" not in kwargs:
            raise ValueError("Search manager is required for code search tool")
        self.search_manager = kwargs.get("search_manager")

    class ACRSearchMethodInClassInput(BaseModel):
        method_name: str = Field(description="Name of the method to search for", examples=["get_user_by_id"])
        class_name: str = Field(description="Name of the class to search in", examples=["MyClass"])

    name = "search-method-in-class"
    description = """
    Search for a method in a given class.
    """

    args_schema = ACRSearchMethodInClassInput

    def _run(self, method_name: str, class_name: str) -> str:
        return self.search_manager.search_method_in_class(method_name, class_name)[0]

    @property
    def short_name(self) -> Optional[str]:
        return "acr"

    search_manager: Any = None


class ACRSearchMethod(CEBaseTool):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args_schema = self.ACRSearchMethodInput
        if self.dry_run:
            return
        if "search_manager" not in kwargs:
            raise ValueError("Search manager is required for code search tool")
        self.search_manager = kwargs.get("search_manager")

    class ACRSearchMethodInput(BaseModel):
        method_name: str = Field(description="Name of the method to search for", examples=["get_user_by_id"])

    name = "search-method"
    description = """
    Search for a method in the entire codebase.
    """

    args_schema = ACRSearchMethodInput

    def _run(self, method_name: str) -> str:
        return self.search_manager.search_method(method_name)[0]

    @property
    def short_name(self) -> Optional[str]:
        return "acr"

    search_manager: Any = None


class ACRSearchCode(CEBaseTool):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args_schema = self.ACRSearchCodeInput
        if self.dry_run:
            return
        if "search_manager" not in kwargs:
            raise ValueError("Search manager is required for code search tool")
        self.search_manager = kwargs.get("search_manager")

    class ACRSearchCodeInput(BaseModel):
        code_str: str = Field(description="Code snippet to search for", examples=["import os"])

    name = "search-code"
    description = """
    Search for a code snippet in the entire codebase.
    """

    args_schema = ACRSearchCodeInput

    def _run(self, code_str: str) -> str:
        return self.search_manager.search_code(code_str)[0]

    @property
    def short_name(self) -> Optional[str]:
        return "acr"

    search_manager: Any = None


class ACRSearchCodeInFile(CEBaseTool):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args_schema = self.ACRSearchCodeInFileInput
        if self.dry_run:
            return
        if "search_manager" not in kwargs:
            raise ValueError("Search manager is required for code search tool")
        self.search_manager = kwargs.get("search_manager")

    class ACRSearchCodeInFileInput(BaseModel):
        code_str: str = Field(description="Code snippet to search for", examples=["import os"])
        file_path: str = Field(description="Path to the file to search in", examples=["path/to/file.py"])

    name = "search-code-in-file"
    description = """
    Search for a code snippet in a given file.
    """

    args_schema = ACRSearchCodeInFileInput

    def _run(self, code_str: str, file_path: str) -> str:
        return self.search_manager.search_code_in_file(code_str, file_path)[0]

    @property
    def short_name(self) -> Optional[str]:
        return "acr"

    search_manager: Any = None
