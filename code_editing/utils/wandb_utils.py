import time
from typing import Dict, List

import wandb

from code_editing.code_editor import CEInput


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
    return {
        "instruction": req["instruction"],
        "code_base_beautified": "\n".join(
            [f"* `{file_name}`\n```\n{file_contents}\n```" for file_name, file_contents in req["code_base"].items()]
        ),
    }
