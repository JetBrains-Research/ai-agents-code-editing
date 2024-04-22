import ast
import logging
from textwrap import dedent
from typing import Any, Dict

from code_editing.data_sources.base_source import CEDataSource
from code_editing.data_sources.hf_source import HuggingFaceSimpleGitCEDataSource
from code_editing.metrics.base_metric import BaseSentenceMetric
from code_editing.metrics.utils import extract_patch
from code_editing.utils.git_utils import apply_patch_like_commit, get_changed_files_patch


def as_any(x) -> Any:
    return x


def normalize_code_by_ast(code: str, sort_keyargs: bool = True, remove_doc_string: bool = True) -> str:
    """Normalize the code by parsing and unparsing it using the AST module.
    If parsing fails, return the original code."""

    class KeyargSorter(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call):
            if node.keywords:
                node.keywords.sort(key=lambda x: x.arg or "None")
            return node

    class DocStringremover(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef):
            return self._visit_def(node)

        def visit_Module(self, node: ast.Module) -> Any:
            return self._visit_def(node)

        def visit_ClassDef(self, node: ast.ClassDef):
            return self._visit_def(node)

        def _visit_def(self, node):
            node = as_any(self.generic_visit(node))
            match node.body:
                case [ast.Expr(value=ast.Constant(value=str())), *body]:
                    node.body = body
            return node

    try:
        tree = ast.parse(dedent(code))
        if remove_doc_string:
            tree = DocStringremover().visit(tree)
        if sort_keyargs:
            tree = KeyargSorter().visit(tree)
        return ast.unparse(tree)
    except (SyntaxError, ValueError):
        return code


def code_equal(code1: str, code2: str) -> bool:
    """
    Compares two python code fragments comparing them by AST

    Adapted from CoEditor: https://github.com/MrVPlusOne/Coeditor/blob/main/src/coeditor/common.py
    """
    if code1 == code2:
        return True
    code1 = normalize_code_by_ast(code1)
    code2 = normalize_code_by_ast(code2)
    return code1 == code2


class ExactMatchMetric(BaseSentenceMetric):
    def __init__(self, data_source: CEDataSource, **kwargs):
        if not isinstance(data_source, HuggingFaceSimpleGitCEDataSource):
            raise ValueError("ExactMatchMetric can only be used with HuggingFaceSimpleGitCEDataSource")
        self.data_source: HuggingFaceSimpleGitCEDataSource = data_source

    def _score_single(self, diff_true: str, diff_pred: str, full_row: Dict):
        patch = extract_patch(diff_pred)
        if patch is None:
            return 0
        diff_true = diff_true + "\n"
        # Get the files that were changed in the commit
        files = get_changed_files_patch(full_row["repo"], diff_true, self.data_source.data_path, full_row["base_hash"])
        # Get the file contents with the patch applied to the base commit
        ctx_pred = apply_patch_like_commit(
            full_row["repo"], full_row["base_hash"], patch, files, self.data_source.data_path
        )
        if ctx_pred is None:
            return 0
        # Get the file contents with the true modifications applied to the base commit
        ctx_true = apply_patch_like_commit(
            full_row["repo"], full_row["base_hash"], diff_true, files, self.data_source.data_path
        )
        if ctx_true is None:
            logging.warning(f"Error in getting true context for {full_row['repo']} {full_row['base_hash']}")
            return 0
        # Compare the two
        for key in ctx_true:
            code_true = ctx_true[key]
            code_pred = ctx_pred[key]

            if not code_equal(code_true, code_pred):
                return 0
        return 1
