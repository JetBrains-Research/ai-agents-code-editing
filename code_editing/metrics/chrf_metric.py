import logging
from typing import List

import evaluate

from code_editing.metrics.base_metric import BaseMetric
from code_editing.metrics.utils import compare_diffs, extract_patches


class ChrfMetric(BaseMetric):
    """
    This class implements the chrF metric from the huggingface evaluate library.
    """

    def __init__(self, **kwargs):
        self.model_name = "chrf"
        self._evaluator = evaluate.load(self.model_name)

    def _score(self, diff_true: List[str], diff_pred: List[str], _):
        res = self._evaluator.compute(references=[[x] for x in diff_true], predictions=extract_patches(diff_pred))
        return res["score"] / 100


class ChrfFileMetric(BaseMetric):
    """
    This class implements the chrF metric for the file changes level
    """

    def __init__(self, **kwargs):
        self.model_name = "chrf"
        self._evaluator = evaluate.load(self.model_name)

    def _score(self, diff_true: List[str], diff_pred: List[str], _):
        diff_pred = extract_patches(diff_pred)
        files_pred, files_true = [], []
        for d_pred, d_true in zip(diff_pred, diff_true):
            f_pred, f_true = compare_diffs(d_pred, d_true)
            files_pred.extend(f_pred)
            files_true.extend(f_true)
        res = self._evaluator.compute(references=[[x] for x in files_true], predictions=files_pred)
        return res["score"] / 100
