# Original:
import ast
import inspect
import logging
import re
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from code_editing.agents.context_providers.acr_search.search_manage import SearchManager
from code_editing.agents.context_providers.acr_search.search_utils import to_relative_path
from code_editing.agents.graph_factory import GraphFactory
from code_editing.agents.run import RunOverviewManager, ToolUseStatus

SYSTEM_PROMPT = """You are a software developer maintaining a large project.
You are working on an issue submitted to your project.
The issue contains a description marked between <issue> and </issue>.
Your task is to invoke a few search API calls to gather buggy information, then write patches to solve the issues.
"""
prompt = (
    "Based on the files, classes, methods, and code statements from the issue related to the bug, you can use the following search APIs to get more context of the project."
    "\n- search_class(class_name: str): Search for a class in the codebase"
    "\n- search_method_in_file(method_name: str, file_path: str): Search for a method in a given file"
    "\n- search_method_in_class(method_name: str, class_name: str): Search for a method in a given class"
    "\n- search_method(method_name: str): Search for a method in the entire codebase"
    "\n- search_code(code_str: str): Search for a code snippet in the entire codebase"
    "\n- search_code_in_file(code_str: str, file_path: str): Search for a code snippet in a given file file"
    "\n- show_definition(symbol: str, line_no: int, file_path: str): Show the definition of a symbol in a given file"
    "\n\nNote that you can use multiple search APIs in one round."
    "\n\nNow analyze the issue and select necessary APIs to get more context of the project. Each API call must have concrete arguments as inputs."
)
PROXY_PROMPT = """
You are a helpful assistant that retreive API calls and bug locations from a text into json format.
The text will consist of two parts:
1. do we need more context?
2. where are bug locations?
Extract API calls from question 1 (leave empty if not exist) and bug locations from question 2 (leave empty if not exist).

The API calls include:
search_method_in_class(method_name: str, class_name: str)
search_method_in_file(method_name: str, file_path: str)
search_method(method_name: str)
search_class_in_file(self, class_name, file_name: str)
search_class(class_name: str)
search_code_in_file(code_str: str, file_path: str)
search_code(code_str: str)
show_definition(symbol: str, line_no: int, file_path: str)

Provide your answer in JSON structure like this, you should ignore the argument placeholders in api calls.
For example, search_code(code_str="str") should be search_code("str")
search_method_in_file("method_name", "path.to.file") should be search_method_in_file("method_name", "path/to/file")
Make sure each API call is written as a valid python expression.

{
    "API_calls": ["api_call_1(args)", "api_call_2(args)", ...],
    "bug_locations":[{"file": "path/to/file", "class": "class_name", "method": "method_name"}, {"file": "path/to/file", "class": "class_name", "method": "method_name"} ... ]
}

NOTE: a bug location should at least has a "class" or "method".
"""


def parse_function_invocation(
    invocation_str: str,
) -> tuple[str, list[str]]:
    try:
        tree = ast.parse(invocation_str)
        expr = tree.body[0]
        assert isinstance(expr, ast.Expr)
        call = expr.value
        assert isinstance(call, ast.Call)
        func = call.func
        assert isinstance(func, ast.Name)
        function_name = func.id
        raw_arguments = [ast.unparse(arg) for arg in call.args]
        # clean up spaces or quotes, just in case
        arguments = [arg.strip().strip("'").strip('"') for arg in raw_arguments]

        try:
            new_arguments = [ast.literal_eval(x) for x in raw_arguments]
            if new_arguments != arguments:
                pass
        except Exception as e:
            pass
    except Exception as e:
        raise ValueError(f"Invalid function invocation: {invocation_str}") from e

    return function_name, arguments


def is_valid_response(data: Any) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "Json is not a dict"

    if not data.get("API_calls"):
        bug_locations = data.get("bug_locations")
        if not isinstance(bug_locations, list) or not bug_locations:
            return False, "Both API_calls and bug_locations are empty"

        for loc in bug_locations:
            if loc.get("class") or loc.get("method"):
                continue
            return False, "Bug location not detailed enough"
    else:
        for api_call in data["API_calls"]:
            if not isinstance(api_call, str):
                return False, "Every API call must be a string"

            try:
                func_name, func_args = parse_function_invocation(api_call)
            except Exception:
                return False, "Every API call must be of form api_call(arg1, ..., argn)"

            function = getattr(SearchManager, func_name, None)
            if function is None:
                return False, f"the API call '{api_call}' calls a non-existent function"

            arg_spec = inspect.getfullargspec(function)
            arg_names = arg_spec.args[1:]  # first parameter is self

            if len(func_args) != len(arg_names):
                return False, f"the API call '{api_call}' has wrong number of arguments"

    return True, "OK"


def prepare_issue_prompt(problem_stmt: str) -> str:
    """
    Given the raw problem statement, sanitize it and prepare the issue prompt.
    Args:
        problem_stmt (str): The raw problem statement.
            Assumption: the problem statement is the content of a markdown file.
    Returns:
        str: The issue prompt.
    """
    # remove markdown comments
    problem_wo_comments = re.sub(r"<!--.*?-->", "", problem_stmt, flags=re.DOTALL)
    content_lines = problem_wo_comments.split("\n")
    # remove spaces and empty lines
    content_lines = [x.strip() for x in content_lines]
    content_lines = [x for x in content_lines if x != ""]
    problem_stripped = "\n".join(content_lines)
    # add tags
    result = "<issue>" + problem_stripped + "\n</issue>"
    return result


def remove_unwanted_lines(text: str, word: str) -> str:
    return "\n".join([e for e in text.split("\n") if word not in e])


class ACRRetrieval(GraphFactory):
    name = "acr_retrieval"

    def __init__(self, *args, max_tries: int = 5, use_show_definition: bool = False, **kwargs):
        # super().__init__(*args, **kwargs)
        super().__init__()
        self.max_tries = max_tries
        self.prompt = prompt
        self.proxy_prompt = PROXY_PROMPT
        if not use_show_definition:
            # remove corresponding line
            self.prompt = remove_unwanted_lines(self.prompt, "show_definition")
            self.proxy_prompt = remove_unwanted_lines(self.proxy_prompt, "show_definition")

    def proxy_run(self, text: str) -> Optional[dict]:
        messages = [SystemMessage(self.proxy_prompt)]
        messages.append(HumanMessage(text))
        llm: BaseChatModel = self._llm
        parser = JsonOutputParser()

        for i in range(self.max_tries):
            res = llm.invoke(messages)
            messages.append(res)
            output = parser.invoke(res)
            valid, diagnosis = is_valid_response(output)
            if valid:
                return output
            else:
                messages.append(HumanMessage(f"{diagnosis}. Please provide a valid response."))
                continue

        logging.warning("Failed to get a valid response after max tries.")
        return None

    def build(self, *args, run_overview_manager: RunOverviewManager, **kwargs):
        # noinspection PyTypeChecker
        search_manager: SearchManager = run_overview_manager.get_ctx_provider("search_manager")

        workflow = StateGraph(dict)
        llm: BaseChatModel = self._llm
        search_text = self.prompt

        iters = 0

        messages: List[BaseMessage] = [SystemMessage(SYSTEM_PROMPT)]

        def start(state):
            nonlocal messages
            instruction = state["instruction"]
            messages.append(HumanMessage(prepare_issue_prompt(instruction)))
            return state

        def search(state):
            nonlocal messages, iters, search_text
            iters += 1

            for i in range(1, self.max_tries + 1):
                messages.append(HumanMessage(search_text))
                search_text = (
                    "Based on your analysis, answer below questions:\n"
                    "    - do we need more context: construct search API calls to get more context of the project. (leave it empty if you don't need more context)\n"
                    "    - where are bug locations: buggy files and methods. (leave it empty if you don't have enough information)"
                )
                res = llm.invoke(messages)
                messages.append(res)
                output = self.proxy_run(res.content)
                if output is None:
                    messages.append(HumanMessage("Failed to get a valid response. Please provide a valid response."))
                    continue
                api_calls = output.get("API_calls", [])
                bug_locations = output.get("bug_locations", [])

                state["is_sufficient"] = api_calls == [] and bug_locations != []
                state["api_calls"] = api_calls
                state["bug_locations"] = bug_locations

                if bug_locations:
                    collated_tool_response = ""
                    # check bug locations
                    for bug_location in bug_locations:
                        try:
                            tool_output, *_ = search_for_bug_location(search_manager, bug_location)
                        except Exception as e:
                            tool_output = f"Cound not find bug location: {e}"
                        collated_tool_response += f"\n\n{tool_output}\n"

                    if "Unknown function" in collated_tool_response or "Could not" in collated_tool_response:
                        messages.append(
                            HumanMessage(
                                "The buggy locations is not precise. You may need to check whether the arguments are correct and search more information."
                            )
                        )
                        state["is_sufficient"] = False

                return state

        def do_search(state):
            nonlocal messages, llm, run_overview_manager
            api_calls = state["api_calls"]
            if api_calls:
                tool_output = ""
                for api_call in api_calls:
                    try:
                        func_name, func_args = parse_function_invocation(api_call)
                        function = getattr(search_manager, func_name)
                        try:
                            run_overview_manager.log_tool_use(func_name, ToolUseStatus.CALL)
                            res, summary, ok = function(*func_args)
                            if ok:
                                run_overview_manager.log_tool_use(func_name, ToolUseStatus.OK)
                            else:
                                run_overview_manager.log_tool_use(func_name, ToolUseStatus.FAIL)
                        except Exception:
                            run_overview_manager.log_tool_use(func_name, ToolUseStatus.THROWN)
                            raise
                        tool_output += f"Result of {func_name}({', '.join(func_args)}):\n{res}\n"
                    except Exception as e:
                        tool_output += f"Error in {api_call}: {e}\n"
                messages.append(HumanMessage(tool_output))
                messages.append(HumanMessage("Let's analyze collected context first"))
                res = llm.invoke(messages)
                messages.append(res)
            return state

        workflow.add_node("start", RunnableLambda(start, name="start"))
        workflow.add_node("search", RunnableLambda(search, name="search"))
        workflow.add_node("do_search", RunnableLambda(do_search, name="do_search"))

        workflow.add_edge("start", "search")
        workflow.add_conditional_edges(
            "search",
            lambda state: state["is_sufficient"] or iters >= 15,
            {True: END, False: "do_search"},
        )
        workflow.add_edge("do_search", "search")
        workflow.set_entry_point("start")

        def collect_context(state):
            bug_locations = state.get("bug_locations", [])
            search_manager.is_tracking = True
            for bug_location in bug_locations:
                search_for_bug_location(search_manager, bug_location)
            # compile into dict of lines
            segments = search_manager.viewed_lines
            ctx = {}
            for file_name, st, end in segments:
                fname = to_relative_path(file_name, run_overview_manager.repo_path).replace("\\", "/")
                ctx.setdefault(fname, set()).update(range(st, end + 1))
            return {"collected_context": ctx}

        return workflow.compile() | RunnableLambda(collect_context, name="collect_context")


def search_for_bug_location(
    search_manager: SearchManager,
    bug_location: dict[str, str],
) -> tuple[str, str, bool]:
    found = False

    file_name = bug_location.get("file")
    method_name = bug_location.get("method")
    class_name = bug_location.get("class")

    assert method_name or class_name, f"Invalid bug location: {bug_location}"

    call_result = None

    def call_function(func_name: str, kwargs: dict[str, str]) -> None:
        nonlocal found, call_result, search_manager
        func = getattr(search_manager, func_name)
        call_result = func(**kwargs)
        found = True

    if (not found) and method_name and class_name:
        kwargs = {
            "method_name": method_name,
            "class_name": class_name,
        }
        call_function("search_method_in_class", kwargs)

    if (not found) and method_name and file_name:
        kwargs = {
            "method_name": method_name,
            "file_name": file_name,
        }
        call_function("search_method_in_file", kwargs)

    if (not found) and class_name and file_name:
        kwargs = {
            "class_name": class_name,
            "file_name": file_name,
        }
        call_function("search_class_in_file", kwargs)

    if (not found) and class_name:
        kwargs = {"class_name": class_name}
        call_function("get_class_full_snippet", kwargs)

    if (not found) and method_name:
        kwargs = {"method_name": method_name}
        call_function("search_method", kwargs)

    assert call_result

    return call_result
