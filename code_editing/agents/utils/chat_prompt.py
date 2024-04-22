from langchain import hub
from langchain_core.prompts import ChatPromptTemplate


class ChatPromptFactory:
    def __init__(self, owner_repo_commit: str):
        self.owner_repo_commit = owner_repo_commit
        self.prompt = hub.pull(owner_repo_commit)

    def build(self) -> ChatPromptTemplate:
        return self.prompt

    @property
    def name(self):
        return self.owner_repo_commit.split("/")[1]
