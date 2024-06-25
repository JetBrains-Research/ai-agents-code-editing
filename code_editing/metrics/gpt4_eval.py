import logging
import re
from typing import Dict

from openai import OpenAI

from code_editing.metrics.base_metric import BaseSentenceMetric
from code_editing.metrics.utils import extract_patch
from code_editing.utils import wandb_utils


class GPT4EvaluationMetric(BaseSentenceMetric):
    def __init__(self, model_name: str = "gpt-4-0125-preview", **kwargs):
        # Use OpenAI API
        self.openai_api = OpenAI(api_key=kwargs.get("api_key", None))
        self.model_name = model_name
        # Disable OpenAI API logging
        logging.getLogger("httpx").setLevel(logging.WARNING)

    def _score_single(self, diff_true: str, diff_pred: str, full_row: Dict):
        # Get diff
        patch = extract_patch(diff_pred)
        if patch is None:
            patch = diff_pred

        start_ms = wandb_utils.get_current_ms()
        # OpenAI API prompt
        response = self.openai_api.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self.sys_prompt,
                },
                {
                    "role": "user",
                    "content": self.user_prompt_template.format(TRUE=diff_true, PRED=patch),
                },
            ],
            model=self.model_name,
        )
        # Get response
        response = response.choices[0].message.content

        end_ms = wandb_utils.get_current_ms()
        # Parse response from the API
        found = re.findall(r"<score>(.+)</score>", response, re.DOTALL)
        if found:
            try:
                res = float(found[0])
                return res
            except:
                pass
        logging.warning(
            f"GPT4 Evaluation failed to acquire the score from the assessment text. Defaulting to 0...\n\nResponse: {response}"
        )
        return 0

    sys_prompt = """
You will be given two code diffs: the true and the prediction. Your task is to establish how similar the prediction is to the true diff. Please explain the positive and negative aspects of this prediction. Finally, deliver a numerical score for the prediction (from 0 to 10) in corresponding tags (like <score>N</score>).
    """.strip()

    user_prompt_template = """
[true diff start]
{TRUE}
[true diff end]
[prediction diff start]
{PRED}
[prediction diff end]
""".strip()
