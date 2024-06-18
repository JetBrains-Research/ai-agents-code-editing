from typing import Any, Type, Callable

import code_engine_client
from langchain_core.tools import ToolException
from pydantic import BaseModel, ValidationError

from code_editing.agents.context_providers.code_engine import CodeEngineManager
from code_editing.agents.tools.base_tool import CEBaseTool


def hack_pydantic_langchain(T: Type[BaseModel]) -> Type[BaseModel]:
    class FixedModel(T):
        def dict(self, *args, **kwargs):
            return super().dict(by_alias=False, *args, **kwargs)

        @classmethod
        def schema(cls, *args, **kwargs):
            return super().schema(by_alias=False, *args, **kwargs)

    return FixedModel


def tool_from_api(
    func: Callable[[code_engine_client.ApiClient, Any], Any],
    input_type: Type[BaseModel],
    tool_name: str,
    tool_description: str,
    post_process: Callable[[Any], Any] = None,
):
    new_input_type = hack_pydantic_langchain(input_type)

    class ToolFromApi(CEBaseTool):
        name = tool_name
        description = tool_description
        args_schema = new_input_type
        code_engine: CodeEngineManager = None

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.args_schema = new_input_type

            if self.dry_run:
                return

            if self.code_engine is None:
                raise ValueError("CodeEngineManager is required")

        def _run(self, **kwargs) -> Any:
            try:
                inp = input_type.parse_obj(kwargs)
                res = func(self.code_engine.api_client, inp)
                # Post process the result
                if post_process:
                    res = post_process(res)
                # Display the result
                if res is None:
                    return "Success"
                return str(res)
            except ValidationError as e:
                raise ToolException(f"Invalid input: {e}")
            except Exception as e:
                raise ToolException(f"Error in {tool_name}: {e}")

        @property
        def short_name(self) -> str:
            return 'api'

    return ToolFromApi
