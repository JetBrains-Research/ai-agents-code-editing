import os
import tempfile
from typing import Dict, List, Optional

import git
from filelock import FileLock
from git import GitCommandError


def get_repo_path(data_dir, repo) -> str:
    repo = repo.replace("/", "__")
    return str(os.path.join(data_dir, "repos", repo))


def _get_repo(repo_path: str):
    repo = git.Repo(repo_path)
    return repo


def lock_repo(repo_path: str, data_dir: str) -> FileLock:
    """Lock a repo to modify it."""
    # abs path of repo -> lock name
    lock_name = os.path.abspath(repo_path).replace("/", "_").replace("\\", "_")
    lock_name = "".join([c for c in lock_name if c.isalnum() or c in "_-"])
    lock_file = os.path.join(data_dir, "repos", f"{lock_name}.lock")
    return FileLock(lock_file)


def _prep_repo(repo: git.Repo):
    repo.git.fetch("--all")
    repo.git.checkout("HEAD", ".")
    repo.git.clean("-fd")
    with repo.config_writer("repository") as cw:
        cw.set_value("core", "autocrlf", "true")


def _read_files(repo: git.Repo, files: List[str]) -> Dict[str, str]:
    file_contents = {}
    for file in files:
        file_path = os.path.join(repo.working_dir, file)
        if not os.path.exists(file_path):
            # File does not exist (e.g. created in the diff)
            file_contents[file] = ""
            continue
        try:
            with open(file_path, "r", encoding="utf8", errors="ignore") as f:
                file_contents[file] = f.read()
        except Exception as e:
            file_contents[file] = ""
            print(f"Can not read file with ext {file}. Replace with empty string...", e)
    return file_contents


def _fetch_commit(repo: git.Repo, commit_sha: str) -> None:
    repo.remotes.origin.fetch(commit_sha)


def _checkout_commit(repo: git.Repo, commit_sha: str) -> None:
    # Fetch origin
    try:
        repo.git.checkout(commit_sha, force=True)
    except GitCommandError:
        # Try to fetch the commit
        _fetch_commit(repo, commit_sha)
        repo.git.checkout(commit_sha, force=True)


def get_parent_commit_sha(repo: str, commit_sha: str, data_dir: str) -> str:
    """Get the parent commit sha of a commit."""
    repo_path = get_repo_path(data_dir, repo)
    repo = _get_repo(repo_path)
    _prep_repo(repo)
    _fetch_commit(repo, commit_sha)
    parent_commit_sha = repo.commit(commit_sha).parents[0].hexsha
    return parent_commit_sha


def get_repo_spec_content_on_commit(repo: str, base_commit_sha: str, files: List[str], data_dir: str) -> Dict[str, str]:
    repo_path = get_repo_path(data_dir, repo)
    repo = _get_repo(repo_path)

    _prep_repo(repo)
    _checkout_commit(repo, base_commit_sha)
    file_contents = _read_files(repo, files)
    repo.git.checkout("HEAD", ".")

    return file_contents


def apply_patch_like_commit(
    repo: str, base_commit_sha: str, patch: str, files: List[str], data_dir: str
) -> Optional[Dict[str, str]]:
    """Apply a patch to the parent of a git commit. Returns contents of [files] after applying the patch."""
    repo_path = get_repo_path(data_dir, repo)
    repo = _get_repo(repo_path)

    _prep_repo(repo)
    _checkout_commit(repo, base_commit_sha)
    try:
        apply_patch_unsafe(repo, patch)
    except GitCommandError:
        # Failed to apply patch
        repo.git.checkout("HEAD", ".")
        return None
    file_contents = _read_files(repo, files)
    repo.git.checkout("HEAD", ".")

    return file_contents


def clone_repo(repo: str, data_dir: str, repo_url: str = None) -> None:
    """Clone a git repo to data_dir."""
    if repo_url is None:
        repo_url = f"https://github.com/{repo}.git"
    repo_path = get_repo_path(data_dir, repo)
    if os.path.exists(repo_path):
        return
    os.makedirs(repo_path, exist_ok=True)
    repo = git.Repo.clone_from(repo_url, repo_path)
    _prep_repo(repo)


def get_diff(repo: str, commit_sha: str, data_dir: str, base_commit_sha: Optional[str] = None) -> str:
    """Get the diff of a commit."""
    if base_commit_sha is None:
        base_commit_sha = get_parent_commit_sha(repo, commit_sha, data_dir)
    repo_path = get_repo_path(data_dir, repo)
    repo = _get_repo(repo_path)

    _prep_repo(repo)
    _checkout_commit(repo, commit_sha)
    _fetch_commit(repo, base_commit_sha)
    diff = repo.git.diff(base_commit_sha)
    repo.git.checkout("HEAD", ".")

    return str(diff)


def get_changed_files_patch(repo: str, patch: str, data_dir: str, base_commit_sha: str) -> List[str]:
    """Get the changed files of a commit."""
    repo_path = get_repo_path(data_dir, repo)
    repo = _get_repo(repo_path)

    _prep_repo(repo)
    _checkout_commit(repo, base_commit_sha)
    try:
        apply_patch_unsafe(repo, patch)
    except GitCommandError:
        # Failed to apply patch
        repo.git.checkout("HEAD", ".")
        return []
    diff = repo.git.diff(base_commit_sha, name_only=True)
    repo.git.checkout("HEAD", ".")

    return diff.split("\n")


def get_commit_name(repo: str, commit_sha: str, data_dir: str) -> str:
    """Get the commit name of a commit."""
    repo_path = get_repo_path(data_dir, repo)
    repo = _get_repo(repo_path)

    _prep_repo(repo)
    _fetch_commit(repo, commit_sha)
    commit_name = repo.commit(commit_sha).message

    return commit_name


def get_head_diff_unsafe(repo_path: str, _: str) -> str:
    """Get the diff of the head commit."""
    repo = _get_repo(repo_path)
    return repo.git.diff()


def get_head_sha_unsafe(repo_path: str, _: str) -> str:
    """Get the hexsha of the head commit."""
    repo = _get_repo(repo_path)

    return repo.head.commit.hexsha


def checkout_repo(repo: str, commit_sha: str, data_dir: str) -> str:
    """Checkout the repository at given commit and return full path to the directory"""
    repo_path = get_repo_path(data_dir, repo)
    repo = _get_repo(repo_path)

    _prep_repo(repo)
    _checkout_commit(repo, commit_sha)

    return repo_path


def reset_to_head(repo_path: str, _: str) -> None:
    """Reset the repository to the head commit."""
    repo = _get_repo(repo_path)
    _checkout_commit(repo, "HEAD")


def apply_patch_unsafe(repo, patch: str):
    # Temp file for patch contents
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".diff", encoding="utf-8", errors="replace"
    ) as temp_file:
        temp_file.write(patch)
        file_name = temp_file.name
    try:
        # If the patch is empty, do nothing
        if patch.strip() != "":
            # Apply patch
            #  --unidiff-zero: Allow patches with no context
            #  --recount: Fix line numbers in the patch
            repo.git.execute(["git", "apply", "--unidiff-zero", "--recount", "--ignore-whitespace", file_name])
    except Exception as e:
        print(e)
        os.remove(file_name)
        raise e
    os.remove(file_name)
