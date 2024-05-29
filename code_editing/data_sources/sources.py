from typing import Dict, List, Tuple

import pandas as pd

from code_editing.data_sources.git_data import SimpleGitCEData
from code_editing.data_sources.hf_source import HuggingFaceSimpleGitCEDataSource
from code_editing.utils.git_utils import get_parent_commit_sha


class LCACodeEditingDataSource(HuggingFaceSimpleGitCEDataSource):
    def __init__(self, **kwargs):
        super().__init__(
            hub_name="JetBrains-Research/lca-code-editing",
            split="test",
            config="commitchronicle-py-long",
            **kwargs,
        )

    def row_to_data(self, row: Dict) -> SimpleGitCEData:
        base_hash = get_parent_commit_sha(row["repo"], row["hash"], self.data_path)
        return SimpleGitCEData(
            message=row["message"],
            repo=row["repo"],
            diff_true=self._get_diff_helper(row["repo"], row["hash"], base_hash),
            base_hash=get_parent_commit_sha(row["repo"], row["hash"], self.data_path),  # Get parent commit
        )

    def _row_to_repo(self, row: Dict) -> str:
        return row["repo"]


class CIFixPythonDataSource(HuggingFaceSimpleGitCEDataSource):
    def __init__(self, **kwargs):
        super().__init__(
            hub_name="JetBrains-Research/CI-fix-Python",
            split="test",
            **kwargs,
        )

    def row_to_data(self, row: Dict) -> SimpleGitCEData:
        repo = self._row_to_repo(row)
        # Get instruction name
        logs = self._row_to_relevant_logs(row)
        message = f"Fix CI in order for tests to pass. Relevant logs:\n{logs}"
        return SimpleGitCEData(message=message, repo=repo, diff_true=row["diff"], base_hash=row["sha_fail"])

    def _row_to_repo(self, row: Dict) -> str:
        return f'{row["repo_owner"]}/{row["repo_name"]}'

    def _row_to_relevant_logs(self, row: Dict) -> str:
        logs = "\n".join([step["log"] for step in row["logs"]])
        error_index = logs.lower().find("error")
        lines = logs.split("\n")
        # If no error is found, return the last 7 lines
        if error_index == -1:
            return "\n".join(lines[-7:])
        # If error is found, return the 3 lines before and after the error
        line_number = logs[:error_index].count("\n")
        start = max(0, line_number - 3)
        end = min(len(lines), line_number + 4)
        return "\n".join(lines[start:end])


class LCABugLocalizationDataSource(HuggingFaceSimpleGitCEDataSource):
    def __init__(self, config="mixed", **kwargs):
        super().__init__(
            hub_name="JetBrains-Research/lca-bug-localization",
            split="dev",
            config=config,
            **kwargs,
        )

    def row_to_data(self, row: Dict) -> SimpleGitCEData:
        repo = self._row_to_repo(row)
        return SimpleGitCEData(
            repo=repo,
            diff_true=row["diff"],
            base_hash=row["base_sha"],
            message=row["issue_title"],
        )

    def _row_to_repo(self, row: Dict) -> str:
        return f'{row["repo_owner"]}/{row["repo_name"]}'


class SWEBenchDataSource(HuggingFaceSimpleGitCEDataSource):
    def __init__(self, lite: bool = False, split_name: str = "test", **kwargs):
        hub_name = "princeton-nlp/SWE-bench_Lite" if lite else "princeton-nlp/SWE-bench"
        super().__init__(
            hub_name=hub_name,
            split=split_name,
            **kwargs,
        )

    def row_to_data(self, row: Dict) -> SimpleGitCEData:
        return SimpleGitCEData(
            repo=row["repo"],
            diff_true=row["patch"],
            base_hash=row["base_commit"],
            message=row["problem_statement"],
        )

    def _row_to_repo(self, row: Dict) -> str:
        return row["repo"]

    def to_swebench_results(self, inference: pd.DataFrame, model_name: str) -> Tuple[List[dict], List[str]]:
        df = inference.copy()
        ds_df = self._dataset.to_pandas()
        df["diff_pred"] = df["diff_pred"].apply(lambda x: x if not pd.isna(x) else "")
        df["diff_pred"] = df["diff_pred"].apply(lambda x: x if x else "")
        out = df["diff_pred"].to_list()

        def instance_id(i):
            return ds_df[ds_df["base_commit"] == df["base_hash"].iloc[i]]["instance_id"].iloc[0]

        out = [
            {"instance_id": instance_id(i), "model_patch": out[i], "model_name_or_path": model_name}
            for i in range(len(out))
        ]
        instance_ids = [e["instance_id"] for e in out]
        return out, instance_ids
