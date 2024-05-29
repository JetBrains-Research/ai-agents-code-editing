import logging
import re
from textwrap import dedent
from typing import Any, Dict, List

import pandas as pd
from openai import OpenAI

from code_editing.metrics.base_metric import BaseSentenceMetric
from code_editing.metrics.utils import extract_patch


class GPT4ComparisonMetric(BaseSentenceMetric):
    def __init__(self, second_path: str, model_name: str = "gpt-4-0125-preview", **kwargs):
        # Use OpenAI API
        self.openai_api = OpenAI(api_key=kwargs.get("api_key", None))
        self.model_name = model_name
        self.second_pred = pd.read_json(second_path, lines=True)
        # Disable OpenAI API logging
        logging.getLogger("httpx").setLevel(logging.WARNING)

    def _score_single(self, diff_true: str, diff_pred: str, full_row: Dict):
        diff_pred1 = diff_pred
        # Find the second prediction
        diff_pred2 = self.second_pred[self.second_pred["base_hash"] == full_row["base_hash"]]["diff_pred"].values[0]

        patch1 = extract_patch(diff_pred1)
        if patch1 is None:
            patch1 = diff_pred1 or ""
        patch2 = extract_patch(diff_pred2)
        if patch2 is None:
            patch2 = diff_pred2 or ""

        max_chars = 30000 * 4
        patch1 = patch1[:max_chars]
        patch2 = patch2[:max_chars]

        # OpenAI API prompt
        response = self.openai_api.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self.sys_prompt,
                },
                {
                    "role": "user",
                    "content": self.user_prompt_template.format(TRUE=diff_true, PRED1=patch1, PRED2=patch2),
                },
            ],
            model=self.model_name,
        )
        # Get response
        response = response.choices[0].message.content
        # Parse response from the API
        found = re.findall(r"<score>(.+)</score>", response, re.DOTALL)
        if found and len(found) == 1:
            try:
                res = float(found[0])
                return res
            except:
                pass
        logging.warning(
            f"GPT4 Evaluation failed to acquire the score from the assessment text. Defaulting to 0...\n\nResponse: {response}"
        )
        return 0.5

    sys_prompt = dedent(
        """As an expert in Git and code editing, you are tasked with analyzing and comparing three code diffs: Prediction 1 diff, Prediction 2 diff, and the ground truth diff.### Your goal is to assess the similarity of each prediction to the ground truth diff and determine which prediction aligns the closest. Provide a detailed evaluation of the positive and negative aspects of each prediction in relation to the true diff.### Finally, render a verdict by assigning a score (1 or 2) to the prediction that performed the best, encapsulating your assessment in corresponding tags (e.g., <score>1</score> or <score>2</score>). Ensure your analysis is thorough and insightful, highlighting key differences and similarities that influence the comparison process. Your expertise in Git diffs will be crucial in delivering a comprehensive and informed evaluation."""
    )

    user_prompt_template = """
[prediction 1 diff start]
{PRED1}
[prediction 1 diff end]
[prediction 2 diff start]
{PRED2}
[prediction 2 diff end]
[true diff start]
{TRUE}
[true diff end]
""".strip()

    def _accum(self, objs: List[Any]):
        wins_1, wins_2, ties = 0, 0, 0
        for obj in objs:
            if obj == 1:
                wins_1 += 1
            elif obj == 2:
                wins_2 += 1
            elif obj == 0.5:
                ties += 1
        invalid = len(objs) - (wins_1 + wins_2 + ties)
        wins_1 /= len(objs)
        wins_2 /= len(objs)
        ties /= len(objs)
        invalid /= len(objs)
        return {"win_1": wins_1, "win_2": wins_2, "tie": ties, "invalid": invalid}
