from typing import Dict

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
        base_hash = get_parent_commit_sha(row["repo"], row["hash"], self._data_path)
        return SimpleGitCEData(
            message=row["message"],
            repo=row["repo"],
            diff_true=self._get_diff(row["repo"], row["hash"], base_hash),
            base_hash=get_parent_commit_sha(row["repo"], row["hash"], self._data_path),  # Get parent commit
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
    def __init__(self, split: str, **kwargs):
        super().__init__(
            hub_name="princeton-nlp/SWE-bench",
            split=split,
            **kwargs,
        )

    def row_to_data(self, row: Dict) -> SimpleGitCEData:
        return SimpleGitCEData(
            repo=row["repo"],
            diff_true=row["patch"],
            base_hash=row["base_commit"],
            message=row["problem_statement"],
        )
