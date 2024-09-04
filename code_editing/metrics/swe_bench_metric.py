import json
import os
import subprocess
import tempfile
import uuid
from typing import List

import pandas as pd

from code_editing.data_sources import SWEBenchDataSource
from code_editing.data_sources.base_source import CEDataSource
from code_editing.metrics.base_metric import BaseMetric


class SWEBenchMetric(BaseMetric):
    """
    This class implements the pass rate metric using the SWE-Bench evaluation harness.
    """

    def __init__(self, data_source: CEDataSource, max_workers: int = 12, cache_level: str = "none", **kwargs):
        self.max_workers = max_workers
        self.cache_level = cache_level

        if not isinstance(data_source, SWEBenchDataSource):
            raise ValueError("SWEBench pass rate calculations can only be made with SWEBenchDataSource")
        self.data_source: SWEBenchDataSource = data_source

    def _score(self, _: List[str], __: List[str], df: pd.DataFrame):
        model_name = df["model_name"].iloc[0] if "model_name" in df.columns else "unknown"
        swebench_obj, instance_ids = self.data_source.to_swebench_results(df, model_name)

        # create a temporary file with the model predictions
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(swebench_obj, f)
            f.close()
            swebench_path = f.name
        unique_hex = uuid.uuid4().hex[:8]

        # We need docker to run the evaluation harness
        #  HACK: To give access to the docker socket, we can use the following command:
        #  sudo chmod 666 /var/run/docker.sock
        cmd = [
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            self.data_source.name,
            "--split",
            self.data_source.split,
            "--predictions_path",
            swebench_path,
            "--max_workers",
            str(self.max_workers),
            "--run_id",
            unique_hex,
            "--cache_level",
            str(self.cache_level),
        ]

        # run the evaluation harness
        subprocess.run(cmd, check=True)

        # read the results
        results_path = f"{model_name}.{unique_hex}.json"
        with open(results_path) as f:
            results = json.load(f)

        # remove the temporary files
        os.remove(swebench_path)
        os.remove(results_path)

        # calculate the pass rate
        total = len(df)
        n_resolved = results.get("resolved_instances", 0)
        results["pass_rate"] = n_resolved / total if total > 0 else 0

        return results
