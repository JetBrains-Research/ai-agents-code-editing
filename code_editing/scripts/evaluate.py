import json
import logging
import os.path

import dotenv
import hydra
import omegaconf
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from tqdm import tqdm

import wandb

dotenv.load_dotenv()

from code_editing.configs.evaluation_config import RunEvaluationConfig
from code_editing.data_sources import SWEBenchDataSource
from code_editing.data_sources.extract_code_base import CodeBaseExtractor
from code_editing.metrics.base_metric import BaseMetric


@hydra.main(version_base=None, config_path="conf", config_name="evaluation")
def main(cfg: RunEvaluationConfig):
    """This script evaluates a csv file with columns diff_true and diff_pred."""
    # Instantiate the extractor and data source
    extractor: CodeBaseExtractor = instantiate(cfg.extractor)
    data_source = instantiate(cfg.data_source, extractor=extractor)

    # Read the input file specified by the user
    if cfg.input_path.endswith(".csv"):
        df = pd.read_csv(cfg.input_path)
    else:
        df = pd.read_json(cfg.input_path, lines=True)
        # if SWE format, convert to the expected format
        if "model_name_or_path" in df.columns and isinstance(data_source, SWEBenchDataSource):
            # rename model_name_or_path to model_name
            dataset_df = data_source._dataset.to_pandas()
            # add diff_true, message, repo, base_hash from the dataset merged on instance_id
            df = df.merge(dataset_df, left_on="instance_id", right_on="instance_id")
            df["model_name"] = df["model_name_or_path"]
            df["diff_pred"] = df["model_patch"]
            df["diff_true"] = df["patch"]
            df["message"] = df["problem_statement"]
            df["base_hash"] = df["base_commit"]
            df["viewed_lines"] = df["diff_true"].apply(lambda x: "{}")
            df = df[["diff_pred", "diff_true", "repo", "base_hash", "message", "viewed_lines", "model_name"]]

    # Get the 'diff_pred' column from the dataframe, replace any NaN values with an empty string
    diff_pred = df["diff_pred"].fillna("")

    # Get the 'diff_true' column from the data source
    diff_true = df["diff_true"]

    # W&B sugary stuff
    model_name = df["model_name"].iloc[0] if "model_name" in df.columns else None
    run_name = model_name or os.path.split(cfg.input_path)[-2]
    tags = []
    # If the model name is in the format <run_name>_hex, split it
    if len(model_name) > 8 and model_name[-9] == "_":
        run_name = model_name[:-9]
        tags.append(f"unique:{model_name[-8:]}")

    res = {}
    pbar = tqdm(cfg.metrics.items(), position=0, desc="Running metrics")
    tags += cfg.metrics.keys()

    use_wandb = cfg.wandb.enable
    if use_wandb:
        config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(project=cfg.wandb.project, job_type="evaluation", config=config, tags=tags)
        wandb.run.name = f"Eval {run_name}"
        if model_name is not None:
            try:
                wandb.use_artifact(f"inference_results.{model_name}:latest")
            except:
                logging.warning(f"Failed to use the artifact inference_results.{model_name}:latest", exc_info=True)

    # Lock all repositories in the data source
    locks = data_source.all_locks()
    for lock in locks:
        lock.acquire()

    # Iterate through metrics
    for name, metric_conf in pbar:
        pbar.set_postfix_str(name)
        try:
            metric: BaseMetric = instantiate(metric_conf, data_source=data_source)
            res[name] = metric.score(diff_true, diff_pred, df, n_samples=metric_conf.get("limit_count", None))
            if use_wandb:
                wandb.log({name: res[name]})
        except Exception as e:
            logging.error(f"Failed to run metric {name}", exc_info=e)
            res[name] = None

    # Release all locks in the data source
    for lock in locks:
        lock.release()

    res_json = json.dumps(res, indent=2)
    print(res_json)
    # Save the results to the output path
    hydra_output_path = os.path.join(HydraConfig.get().runtime.output_dir, "eval.json")
    with open(hydra_output_path, "w") as f:
        f.write(res_json)
    if use_wandb:
        wandb.log_artifact(hydra_output_path, name=f"evaluation_results.{model_name}", type="evaluation_results")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
