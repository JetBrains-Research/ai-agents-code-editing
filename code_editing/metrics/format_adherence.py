from typing import Dict

from code_editing.data_sources.base_source import CEDataSource
from code_editing.data_sources.hf_source import HuggingFaceSimpleGitCEDataSource
from code_editing.metrics.base_metric import BaseSentenceMetric
from code_editing.metrics.utils import extract_patch
from code_editing.utils.git_utils import apply_patch_like_commit


class FormatAdherenceWeak(BaseSentenceMetric):
    def __init__(self, **kwargs):
        pass

    def _score_single(self, _: str, diff_pred: str, __):
        patch = extract_patch(diff_pred)
        if patch is not None:
            return 1
        else:
            return 0


class SuccessfulGeneration(BaseSentenceMetric):
    def __init__(self, **kwargs):
        pass

    def _score_single(self, _: str, diff_pred: str, __):
        return diff_pred.strip() != ""


class ValidGitDiff(BaseSentenceMetric):
    def __init__(self, data_source: CEDataSource, **kwargs):
        if not isinstance(data_source, HuggingFaceSimpleGitCEDataSource):
            raise ValueError("This metric only works with HuggingFaceSimpleGitCEDataSource")
        self.data_path = data_source.data_path

    def _score_single(self, diff_true: str, diff_pred: str, full_row: Dict):
        patch = extract_patch(diff_pred)
        if patch is None:
            return 0
        # Get the files that were changed in the commit
        #  If successful, this means that the patch is valid
        new_files_contents = apply_patch_like_commit(full_row["repo"], full_row["base_hash"], patch, [], self.data_path)
        if new_files_contents is not None:
            return 1
        else:
            return 0
