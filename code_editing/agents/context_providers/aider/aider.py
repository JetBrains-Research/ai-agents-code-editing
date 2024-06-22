from code_editing.agents.context_providers.aider.repo_map import RepoMap, find_src_files
from code_editing.agents.context_providers.context_provider import ContextProvider


class AiderRepoMap(ContextProvider):
    def __init__(self, repo_path: str, data_path: str):
        self.repo_path = repo_path
        self.data_path = data_path

        self.rm = RepoMap(
            map_tokens=1024,
            root=repo_path,
            token_count=lambda x: len(x.split()) // 4,
        )

    def get_repo_map(self) -> str:
        fnames = find_src_files(self.repo_path)
        repo_map = self.rm.get_repo_map([], fnames)
        return repo_map
