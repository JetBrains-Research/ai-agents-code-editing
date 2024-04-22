import re
from abc import ABC, abstractmethod
from typing import List, Dict

from code_editing.data_sources.git_data import SimpleGitCEData
from code_editing.utils.git_utils import get_repo_spec_content_on_commit, get_diff, get_changed_files_patch


class CodeBaseExtractor(ABC):
    @abstractmethod
    def __call__(self, data: SimpleGitCEData, data_path: str) -> Dict[str, str]:
        pass

    def data_to_files(self, data: SimpleGitCEData, data_path: str) -> List[str]:
        return get_changed_files_patch(data.repo, data.diff_true, data_path, data.base_hash)


class FullFileExtractor(CodeBaseExtractor):
    def __call__(self, data: SimpleGitCEData, data_path: str) -> Dict[str, str]:
        files = self.data_to_files(data, data_path)
        return get_repo_spec_content_on_commit(data.repo, data.base_hash, files, data_path)


class CodeFragmentExtractor(CodeBaseExtractor):
    """
    This class is a subclass of CodeBaseExtractor and is used to extract code fragments from a given data source.
    """

    def __call__(self, data: SimpleGitCEData, data_path: str) -> Dict[str, str]:
        """
        This method is used to extract code fragments from the given data source.
        """

        # Get the list of changed files
        files = self.data_to_files(data, data_path)

        # Get the diff between the current and base commit
        diff = data.diff_true

        # Get the content of the repository at the base commit
        ctx = get_repo_spec_content_on_commit(data.repo, data.base_hash, files, data_path)

        # Find all blocks in the diff that represent changes in the code
        blocks = re.findall(
            r"diff --git .*?\n-{3} a/(.*?)\n.*?@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*?\n", diff, flags=re.DOTALL
        )

        fragments = {}

        # For each block, extract the code fragment
        for block in blocks:
            file, start, length, _, _ = block
            start = int(start)  # Start line
            length = int(length) if length != "" else 1  # Length of the fragment
            end = start + length - 1

            file_content = ctx.get(file, None)
            if file_content is None:
                continue

            fragment_name = f"{file}#L{start}"
            fragment = file_content.split("\n")[start - 1 : end]
            fragment = "\n".join(fragment)

            fragments[fragment_name] = fragment

        return fragments
