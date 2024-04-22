from typing import List

import code_bert_score

from code_editing.metrics.base_metric import BaseMetric
from code_editing.metrics.utils import extract_patches, compare_diffs


class CodeBertScoreMetric(BaseMetric):
    """
    This class implements the CodeBertScore metric.

    It is a wrapper around the code_bert_score library.
    """
    def __init__(self, lang: str = "python", **kwargs):
        self._lang = lang

    def _score(self, diff_true: List[str], diff_pred: List[str], _):
        scores = code_bert_score.score(cands=extract_patches(diff_pred), refs=[[x] for x in diff_true], lang=self._lang)
        # Return mean F1
        return scores[2].mean().item()


class CodeBertScoreFileMetric(BaseMetric):
    """
    This class implements the CodeBertScore metric for the file changes level.

    It is a wrapper around the code_bert_score library.
    """
    def __init__(self, lang: str = "python", **kwargs):
        self._lang = lang

    def _score(self, diff_true: List[str], diff_pred: List[str], _):
        diff_pred = extract_patches(diff_pred)
        files_pred, files_true = [], []
        for d_pred, d_true in zip(diff_pred, diff_true):
            f_pred, f_true = compare_diffs(d_pred, d_true)
            files_pred.extend(f_pred)
            files_true.extend([x] for x in f_true)
        scores = code_bert_score.score(cands=files_pred, refs=files_true, lang=self._lang)
        # Return mean F1
        return scores[2].mean().item()
