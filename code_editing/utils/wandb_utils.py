import time
from typing import List, Dict, Optional, Callable, Any

import wandb
from wandb.sdk.data_types.trace_tree import Trace, StatusCode

from code_editing.backbones.base_backbone import CEInput


def get_current_ms() -> int:
    """Returns the current time in milliseconds."""
    return int(round(time.time() * 1000))


def is_run_active() -> bool:
    """Returns whether the W&B run is active."""
    return wandb.run is not None


def chat_to_dict(
        preprocessed_inputs: List[Dict[str, str]],
) -> Dict[str, str]:
    counters = {}
    res = {}
    for message in preprocessed_inputs:
        if message["role"] == "system":
            key = "system"
        else:
            i = counters.get(message["role"], 0)
            counters[message["role"]] = i + 1
            key = f"{message['role']}_{i}"
        res[key] = message["content"]
    return res


def req_beautify(req: CEInput) -> dict:
    return {"instruction": req["instruction"], "code_base_beautified": '\n'.join(
        [f"* `{file_name}`\n```\n{file_contents}\n```" for file_name, file_contents in req["code_base"].items()])}


def log_prompt_trace(
        parent_span: Trace,
        metadata: Optional[Dict[str, str]] = None,
):
    def wrapper(func: Callable[[CEInput], Any]):
        def new_func(req: CEInput):
            start_ms = get_current_ms()
            res = func(req)
            end_ms = get_current_ms()
            if not is_run_active():
                return res
            # Log to W&B
            if isinstance(res, str):
                chat = {"prompt": res}
            else:
                chat = chat_to_dict(res)
            trace = Trace(
                name="Prompt Generation",
                kind="tool",
                status_code="success",
                start_time_ms=start_ms,
                end_time_ms=end_ms,
                inputs=req_beautify(req),
                outputs=chat,
                model_dict=metadata,
            )
            parent_span.add_child(trace)
            return res

        return new_func

    return wrapper


def log_llm_trace(
        parent_span: Trace,
        model_name: str,
        metadata: Optional[Dict[str, str]] = None,
):
    def wrapper(func):

        def new_func(input_obj):
            start_ms = get_current_ms()
            output_str = func(input_obj)
            end_ms = get_current_ms()
            if not is_run_active():
                return output_str
            if isinstance(input_obj, str):
                input_dict = {"input": input_obj}
            else:
                input_dict = chat_to_dict(input_obj)
            trace = Trace(
                name=f"{model_name} Inference",
                kind="llm",
                status_code="success",
                start_time_ms=start_ms,
                end_time_ms=end_ms,
                inputs=input_dict,
                outputs={"output": output_str},
                model_dict=metadata,
            )
            parent_span.add_child(trace)
            return output_str

        return new_func

    return wrapper


def build_main_trace(
        req: CEInput,
        start_ms: int,
        name: str,
        metadata: Optional[Dict[str, str]] = None,
):
    return Trace(
        name=f"Code Editing: {name}",
        kind="agent",
        status_code="success",
        start_time_ms=start_ms,
        inputs=req_beautify(req),
        model_dict=metadata,
    )


def log_preprocessor_trace(
        before: CEInput,
        after: CEInput,
        start_ms: int,
        end_ms: int,
        parent_span: Trace,
):
    if not is_run_active():
        return
    trace = Trace(
        name="Preprocessor",
        kind="tool",
        status_code="success",
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        inputs=req_beautify(before),
        outputs=req_beautify(after),
    )
    parent_span.add_child(trace)


def log_main_trace(root_span, old_req, resp, status_code, status_message=None):
    if not is_run_active():
        return
    root_span.add_inputs_and_outputs(inputs=req_beautify(old_req), outputs=resp)
    root_span._span.status_code = status_code
    root_span._span.status_message = status_message
    root_span._span.end_time_ms = get_current_ms()
    root_span.log("Code Editing")


def gpt4_eval_trace(
        diff_true: str,
        diff_pred: str,
        start_ms: int,
        end_ms: int,
        score_text: str,
        score_value: Optional[float],
        metadata: Optional[Dict[str, str]] = None,
):
    if not is_run_active():
        return
    trace = Trace(
        name="GPT4 Evaluation",
        kind="llm",
        status_code="success",
        start_time_ms=start_ms,
        end_time_ms=end_ms,
        inputs={"diff_true": diff_true, "diff_pred": diff_pred},
        outputs={"score_text": score_text, "score_value": score_value},
        model_dict=metadata,
    )
    trace.log("GPT4 Evaluation")


def log_codeeditor_trace():
    def wrapper(func):
        def new_func(*args):
            req = args[-1]
            metadata = {}
            if len(args) > 1:
                code_editor = args[0]  # self
                metadata.update(code_editor.metadata)
            start_ms = get_current_ms()
            trace = build_main_trace(req, start_ms, func.__name__, metadata)
            try:
                res = func(*args, root_span=trace)
                err = None
            except Exception as e:
                res = None
                err = e
            if not is_run_active():
                if err:
                    raise err
                return res
            log_main_trace(trace, req, res, StatusCode.SUCCESS if res else StatusCode.ERROR, str(err) if err else None)
            trace.log("Code Editor")
            if err:
                raise err
            return res

        return new_func

    return wrapper
