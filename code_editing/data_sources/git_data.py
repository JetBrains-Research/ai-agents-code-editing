from dataclasses import dataclass


@dataclass
class SimpleGitCEData:
    """
    Dataclass representing a unified interface for all HuggingFace datasets that contain commits from git repositories.

    Attributes:
        message (str): Modification message or instruction.
        repo (str): Repository name. For example, 'keras-team/keras'.
        base_hash (str): Commit hash of the base commit. This is the code before the changes, usually the parent commit.
        diff_true (str): The true diff of the modification.
    """

    message: str
    repo: str
    base_hash: str
    diff_true: str
