import code_engine_client

from code_editing.agents.context_providers.code_engine.api_tool import tool_from_api

ASTGetFileFunctionNames = tool_from_api(
    lambda api, inp: code_engine_client.AstApiApi(api).ast_get_file_functions_names_post(inp),
    code_engine_client.AstApiGetFileFunctionsNamesRequest,
    "get-file-functions",
    "Get functions from a file",
)
ASTGetFileFunctionCode = tool_from_api(
    lambda api, inp: code_engine_client.AstApiApi(api).ast_get_file_function_code_post(inp),
    code_engine_client.AstApiGetFileFunctionCodeRequest,
    "get-file-functions-code",
    "Get code of functions from a file",
)
