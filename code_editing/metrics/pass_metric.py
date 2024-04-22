import json
import logging
import tempfile
import time
from dataclasses import dataclass
from typing import List, Any, Optional

import pandas as pd
from git import GitCommandError
from omegaconf import OmegaConf

from lca.ci_fix_benchmark.benchmark import CIFixBenchmark
from code_editing.data_sources import CIFixPythonDataSource
from code_editing.data_sources.base_source import CEDataSource
from code_editing.metrics.base_metric import BaseMetric
from code_editing.metrics.utils import extract_patch
from code_editing.utils.git_utils import apply_patch_unsafe

logger = logging.getLogger(__name__)


@dataclass
class PassMetric(BaseMetric):
    def __init__(
        self,
        cfg: Any,
        data_source: CEDataSource,
        save_ids: Optional[str] = None,
        load_ids: Optional[str] = None,
        **kwargs,
    ):
        self.cfg = cfg
        if not hasattr(self.cfg, "out_folder"):
            self.cfg.out_folder = tempfile.mkdtemp()
        self.cfg_path = self._save_cfg()
        self.data_source = data_source

        if save_ids is not None and load_ids is not None:
            raise ValueError("Only one of save_ids and load_ids can be set")

        self.save_ids = save_ids
        self.load_ids = load_ids

    def _score(self, diff_true: List[str], diff_pred: List[str], dataset: pd.DataFrame):
        if not isinstance(self.data_source, CIFixPythonDataSource):
            raise ValueError("This metric only supports CIFixPythonDataSource")

        unique_hex = hex(hash(tuple(diff_true + diff_pred)))[2:]

        benchmark = CIFixBenchmark(
            f"code-editing-metrics-{unique_hex}",
            self.cfg_path,
            self.cfg.token_gh,
        )

        if self.load_ids is None:
            full_dataset = self.data_source.full_data().to_pandas()
            # Join the dataset with the full dataset by base_hash in dataset and sha_fail in full_dataset
            ci_dataset = dataset.merge(full_dataset, left_on="base_hash", right_on="sha_fail")

            # Add id, repo_name, repo_owner and sha_fail columns to the dataset
            ci_dataset_dicts = ci_dataset.to_dict(orient="records")

            def fix_repo_function(datapoint, __, repo, ___):
                patch = extract_patch(datapoint["diff_pred"]) or ""
                try:
                    apply_patch_unsafe(repo, patch)
                except GitCommandError:
                    logger.warning(f"Failed to apply patch for repo {repo}")

            # Run the benchmark
            job_ids = benchmark.run_dataset(fix_repo_function, ci_dataset_dicts)

            if self.save_ids is not None:
                with open(self.save_ids, "w") as f:
                    json.dump(job_ids, f)
                logger.info(
                    "Job IDs saved. Please run the metric again with load_ids set to the path of the saved job IDs"
                )
                return -1

            # Wait for the jobs to upload to GitHub Actions
            time.sleep(5)
        else:
            # Load the job IDs
            with open(self.load_ids, "r") as f:
                job_ids = json.load(f)

        # Get the results (synchronous)
        results = benchmark.eval_jobs(job_ids)

        # Calculate the success rate
        n_success = sum([1 for res in results if res["conclusion"] == "success"])
        return n_success / len(dataset)

    def _save_cfg(self):
        omega_config = OmegaConf.create(self.cfg)

        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
            file_path = tmp_file.name
            # Save the OmegaConf object to the temporary file
            OmegaConf.save(omega_config, file_path)
        # Print the path to the temporary file
        return file_path
