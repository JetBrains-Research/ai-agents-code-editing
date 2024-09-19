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

from code_editing.agents.agent_graph import AgentGraph
from code_editing.agents.context_providers.acr_search.search_manage import SearchManager
from code_editing.agents.context_providers.retrieval.retrieval_helper import RetrievalHelper
from code_editing.agents.tools.common import lines_format_document

SYSTEM_PROMPT = """You are a software developer maintaining a large project.
You are working on an issue submitted to your project.
The issue contains a description marked between <issue> and </issue>.
Your task is to invoke a few search API calls to gather buggy information, then write patches to solve the issues.
"""
prompt = (
    "Based on the files, classes, methods, and code statements from the issue related to the bug, you can use the following search APIs to get more context of the project."
    "\n- search_code(code_str: str): Search for a code snippet in the entire codebase"
    "\n\nNow analyze the issue and select necessary APIs to get more context of the project. Each API call must have concrete arguments as inputs."
)
PROXY_PROMPT = """
You are a helpful assistant that retreive API calls and bug locations from a text into json format.
The text will consist of two parts:
1. do we need more context?
2. where are bug locations?
Extract API calls from question 1 (leave empty if not exist) and bug locations from question 2 (leave empty if not exist).

The API calls include:
search_code(code_str: str)

Provide your answer in JSON structure like this, you should ignore the argument placeholders in api calls.
For example, search_code(code_str="str") should be search_code("str")
Make sure each API call is written as a valid python expression.

{
    "API_calls": ["search_code(args)", "search_code(args)", ...],
    "bug_locations":[{"file": "path/to/file", "lines": [[1, 1], [3, 10], [45, 100]]}, {"file": "another/path/to/file", "lines": [[7, 10], [20, 30]]} ... ]
}

NOTE: a bug location should consist of a "file" and "lines", where "lines" is a list of line ranges (1-indexed, inclusive), e.g. [[1, 1], [3, 10], [45, 100]] for lines 1, 3-10, 45-100.
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
            if loc.get("file") is not None and loc.get("lines") is not None:
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


class RetrievalSearchManager:
    def __init__(self, retrieval_helper):
        self.retrieval_helper = retrieval_helper

    def search_code(self, code_str: str, run_manager=None) -> str:
        docs = self.retrieval_helper.search(code_str, 5, run_manager=run_manager)
        res = "\n\n".join(lines_format_document(doc, repo_path=self.retrieval_helper.repo_path) for doc in docs)
        return f"{len(docs)} results found:\n{res}"


class MyACRRetrieval(AgentGraph):
    name = "my_acr_retrieval"

    def __init__(self, max_tries: int = 5, max_iters: int = 15, **kwargs):
        super().__init__(**kwargs)
        self.max_tries = max_tries
        self.max_iters = max_iters

    def proxy_run(self, text: str) -> Optional[dict]:
        messages = [SystemMessage(PROXY_PROMPT)]
        messages.append(HumanMessage(text))
        llm: BaseChatModel = self.llm
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

    @property
    def _runnable(self):
        retrieval_helper = self.get_ctx_provider(RetrievalHelper)
        search_manager = RetrievalSearchManager(retrieval_helper)
        run_manager = self.run_manager

        workflow = StateGraph(dict)
        llm: BaseChatModel = self.llm

        search_text = prompt

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
                    "    - where are bug locations lines: buggy sections in files with line numbers, e.g. 'lines 13-20 and 45 in file utils.py' (leave it empty if you don't have enough information about relevant files or line numbers)\n"
                    "NOTE: Bug location must include line numbers and file path"
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
                return state
            raise ValueError("Failed to get a valid response after max tries.")

        def do_search(state, run_manager=None):
            nonlocal messages, llm, search_manager
            api_calls = state["api_calls"]
            if api_calls:
                tool_output = ""
                for api_call in api_calls:
                    try:
                        func_name, func_args = parse_function_invocation(api_call)
                        function = getattr(search_manager, func_name)
                        res = function(*func_args, run_manager=run_manager)
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
            lambda state: state["is_sufficient"] or iters >= self.max_iters,
            {True: END, False: "do_search"},
        )
        workflow.add_edge("do_search", "search")
        workflow.set_entry_point("start")

        def collect_context(state):
            bug_locations = state.get("bug_locations", [])
            ctx = {}
            for bug_location in bug_locations:
                file_name, lines = bug_location["file"], bug_location["lines"]
                for st, end in lines:
                    ctx.setdefault(file_name, set()).update(range(st, end + 1))
            return {"collected_context": ctx}

        return workflow.compile() | RunnableLambda(collect_context, name="collect_context")
