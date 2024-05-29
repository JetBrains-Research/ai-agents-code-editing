from typing import Any, Optional

from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda


class PromptWrapper:
    """Utility class for creating a prompt template from a string, file or langsmith."""

    def __init__(
        self,
        template: Optional[str] = None,
        template_file: Optional[str] = None,
        owner_repo_commit: Optional[str] = None,
        overrides: Optional[dict] = None,
    ):
        if template is not None:
            self.prompt_template = PromptTemplate.from_template(template)
        elif template_file is not None:
            self.prompt_template = PromptTemplate.from_file(template_file)
        elif owner_repo_commit is not None:
            self.prompt_template = hub.pull(owner_repo_commit)
        else:
            raise ValueError("Either template or template_file should be provided")
        self.overrides = overrides or {}
        self.item_updater = RunnableLambda(lambda x: {**x, **self.overrides}, name="prompt_utils.partial")

    def format(self, **kwargs) -> Any:
        return self.prompt_template.format(**kwargs)

    def as_runnable(self, to_dict: bool = False, do_update: bool = True) -> Runnable:
        if do_update:
            res = self.item_updater | self.prompt_template
        else:
            res = self.prompt_template
        if to_dict:
            res = res | self.str_to_dict
        return res

    str_to_dict = RunnableLambda(lambda x: {"input": x.text})
