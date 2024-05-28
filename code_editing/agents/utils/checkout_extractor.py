import tempfile
from typing import Dict

from code_editing.data_sources.extract_code_base import CodeBaseExtractor
from code_editing.utils.git_utils import clone_repo, checkout_repo


class CheckoutExtractor(CodeBaseExtractor):
    """Utility data extractor for use with the agent module."""

    REPO_KEY = "__REPOPATH__"

    def __init__(self, use_temp_dirs=False):
        self.tmp_dirs = []
        self.use_temp_dirs = use_temp_dirs

    def __call__(self, data, data_path) -> Dict[str, str]:
        if self.use_temp_dirs:
            tmp_data_path = tempfile.mkdtemp()
            self.tmp_dirs.append(tmp_data_path)
            clone_repo(data.repo, tmp_data_path)
            repo_path = checkout_repo(data.repo, data.base_hash, tmp_data_path)
        else:
            repo_path = checkout_repo(data.repo, data.base_hash, data_path)
        return {self.REPO_KEY: repo_path}

    # def __del__(self):
    #     time.sleep(5)  # Wait for git to release the files
    #     for tmp_dir in self.tmp_dirs:
    #         shutil.rmtree(tmp_dir)
