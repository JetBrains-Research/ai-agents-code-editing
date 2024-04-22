import json
import logging
from abc import ABC
from typing import List, Any

import jedi
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score

from code_editing.data_sources.base_source import CEDataSource
from code_editing.data_sources.hf_source import HuggingFaceSimpleGitCEDataSource
from code_editing.metrics.base_metric import BaseSentenceMetric
from code_editing.metrics.utils import extract_patch, edited_lines_per_file
from code_editing.utils.git_utils import get_repo_spec_content_on_commit
from code_editing.utils.tokenization_utils import TokenizationUtils


class BaseLocalizationMetric(BaseSentenceMetric, ABC):
    """
    Base class for localization metrics. It calculates precision, recall, and F1 score from the binary classification results.
    """

    def __init__(self, add_sample_metrics: bool = False, **kwargs):
        self.add_sample_metrics = add_sample_metrics

    def _accum(self, objs: List[Any]):
        recall, precision, f1 = [], [], []
        for obj in objs:
            recall.append(recall_score(obj[0], obj[1], average="binary"))
            precision.append(precision_score(obj[0], obj[1], average="binary"))
            f1.append(f1_score(obj[0], obj[1], average="binary"))

        res = {"recall": np.mean(recall), "precision": np.mean(precision), "f1": np.mean(f1)}

        if self.add_sample_metrics:
            res.update({"sample_recall": recall, "sample_precision": precision, "sample_f1": f1})

        return res

    def to_binary(self, true, pred):
        true_positive = len(true & pred)
        false_positive = len(pred - true)
        false_negative = len(true - pred)
        file_y_true = [1] * (true_positive + false_negative) + [0] * false_positive
        file_y_pred = [1] * true_positive + [0] * false_negative + [1] * false_positive
        return file_y_true, file_y_pred

    def get_viewed_lines(self, full_row):
        return json.loads(full_row.get("viewed_lines", None) or "{}")

    def get_files(self, edited_lines, _, is_viewed):
        y_true, y_pred = edited_lines.keys(), is_viewed.keys()
        return self.to_binary(y_true, y_pred)

    def get_lines(self, edited_lines, _, is_viewed):
        y_true, y_pred = [], []

        for file in set(edited_lines.keys()).union(set(is_viewed.keys())):
            # For each file, we compare the edited lines
            true_lines = set(edited_lines.get(file, []))
            pred_lines = set(is_viewed.get(file, []))

            # Convert the edited lines to binary classification
            file_y_true, file_y_pred = self.to_binary(true_lines, pred_lines)

            y_true.extend(file_y_true)
            y_pred.extend(file_y_pred)

        return y_true, y_pred

    def get_scopes(self, edited_lines, full_row, is_viewed):
        # Get the content of the files on the base commit
        filtered_edited = [file for file in edited_lines.keys() if file.endswith(".py")]
        filtered_viewed = [file for file in is_viewed.keys() if file.endswith(".py")]
        files = list(set(filtered_edited).union(set(filtered_viewed)))
        file_contents = get_repo_spec_content_on_commit(full_row["repo"], full_row["base_hash"], files, self.data_path)

        y_true, y_pred = [], []
        for file in files:
            # For each file, we compare the modified scopes

            # We use jedi to get the context of each line
            script = jedi.Script(file_contents[file])

            # Get the edited lines
            true_lines = set(edited_lines.get(file, []))
            pred_lines = set(is_viewed.get(file, []))

            # Get the set of scopes for the edited lines
            def get_line(line):
                try:
                    return script.get_context(line)
                except Exception as e:
                    print(f"Failed to get py scope at {file}:{line} for {full_row['repo']}@{full_row['base_hash']}", e)
                    return -1

            true_scopes = set([get_line(line) for line in true_lines])
            pred_scopes = set([get_line(line) for line in pred_lines])

            # Convert the scopes to binary classification
            file_y_true, file_y_pred = self.to_binary(true_scopes, pred_scopes)

            y_true.extend(file_y_true)
            y_pred.extend(file_y_pred)

        return y_true, y_pred


class FileEditLocalizationMetric(BaseLocalizationMetric):
    """
    Metric for file-level editing localization.

    It compares the modified files (of old version) in the true and predicted diffs.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _score_single(self, diff_true: str, diff_pred: str, full_row):
        patch = extract_patch(diff_pred)
        if patch is None:
            return [], []

        # Extract the edited lines from the diffs
        true_edited_lines = edited_lines_per_file(diff_true)
        pred_edited_lines = edited_lines_per_file(diff_pred)

        return self.get_files(true_edited_lines, full_row, pred_edited_lines)


class FileViewLocalizationMetric(BaseLocalizationMetric):
    """
    Metric for file-level localization.

    It compares the *viewed* files against the modified files.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _score_single(self, diff_true: str, _: str, full_row):
        viewed_lines = self.get_viewed_lines(full_row)

        # Extract the edited lines from the diffs
        true_edited_lines = edited_lines_per_file(diff_true)

        return self.get_files(true_edited_lines, full_row, viewed_lines)


class LineEditLocalizationMetric(BaseLocalizationMetric):
    """
    Metric for line-level editing localization.

    It compares the modified lines (of old version) in the true and predicted diffs.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _score_single(self, diff_true: str, diff_pred: str, full_row):
        patch = extract_patch(diff_pred)
        if patch is None:
            return [], []

        # Extract the edited lines from the diffs
        true_edited_lines = edited_lines_per_file(diff_true)
        pred_edited_lines = edited_lines_per_file(diff_pred)

        return self.get_lines(true_edited_lines, full_row, pred_edited_lines)


class LineViewLocalizationMetric(BaseLocalizationMetric):
    """
    Metric for line-level localization.

    It compares the *viewed* lines against the modified lines.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _score_single(self, diff_true: str, _, full_row):
        viewed_lines = self.get_viewed_lines(full_row)

        # Extract the edited lines from the diffs
        true_edited_lines = edited_lines_per_file(diff_true)

        return self.get_lines(true_edited_lines, full_row, viewed_lines)


class PyScopeEditLocalizationMetric(BaseLocalizationMetric):
    """
    Metric for Python scope-level localization.

    It compares the modified scopes (i.e. functions, classes, etc.) of the old version in the true and predicted diffs.
    """

    def __init__(self, data_source: CEDataSource, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(data_source, HuggingFaceSimpleGitCEDataSource):
            raise ValueError("This metric only works with HuggingFaceSimpleGitCEDataSource")
        self.data_path = data_source.data_path

    def _score_single(self, diff_true: str, diff_pred: str, full_row):
        patch = extract_patch(diff_pred)
        if patch is None:
            return [], []

        # Extract the edited lines from the diffs
        true_edited_lines = edited_lines_per_file(diff_true)
        pred_edited_lines = edited_lines_per_file(diff_pred)

        return self.get_scopes(true_edited_lines, full_row, pred_edited_lines)


class PyScopeViewLocalizationMetric(BaseLocalizationMetric):
    """
    Metric for Python scope-level localization.

    It compares the *viewed* scopes (i.e. functions, classes, etc.) against the modified scopes.
    """

    def __init__(self, data_source: CEDataSource, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(data_source, HuggingFaceSimpleGitCEDataSource):
            raise ValueError("This metric only works with HuggingFaceSimpleGitCEDataSource")
        self.data_path = data_source.data_path

    def _score_single(self, diff_true: str, _, full_row):
        viewed_lines = self.get_viewed_lines(full_row)

        # Extract the edited lines from the diffs
        true_edited_lines = edited_lines_per_file(diff_true)

        return self.get_scopes(true_edited_lines, full_row, viewed_lines)


class TotalContextMetric(BaseSentenceMetric):
    """
    Metric for the total context size.
    """

    def __init__(self, data_source: CEDataSource, add_sample_metrics: bool = False, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(data_source, HuggingFaceSimpleGitCEDataSource):
            raise ValueError("This metric only works with HuggingFaceSimpleGitCEDataSource")
        self.data_path = data_source.data_path
        self.add_sample_metrics = add_sample_metrics
        self.tok_utils = TokenizationUtils("gpt-3.5-turbo")

    def _score_single(self, diff_true: str, diff_pred: str, full_row):
        viewed_lines = json.loads(full_row.get("viewed_lines", None) or "{}")
        contents = get_repo_spec_content_on_commit(
            full_row["repo"], full_row["base_hash"], viewed_lines.keys(), self.data_path
        )
        total_context = 0
        for file, lines in viewed_lines.items():
            code_lines = contents.get(file, None)
            if code_lines is None:
                logging.warning(f"File {file} not found in the repo {full_row['repo']}@{full_row['base_hash']}")
                continue
            code_lines = code_lines.split("\n")
            for line in lines:
                if line <= 0 or line > len(code_lines):
                    logging.warning(
                        f"Line {line} out of bounds in file {file} in the repo {full_row['repo']}@{full_row['base_hash']}"
                    )
                    continue
                total_context += self.tok_utils._count_tokens_completion(code_lines[line - 1])
        return total_context

    def _accum(self, objs: List[Any]):
        res = np.mean(objs)
        if self.add_sample_metrics:
            return {"total_context": res, "sample_total_context": objs}
        return {"total_context": res}
