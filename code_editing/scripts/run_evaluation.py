import json
import logging
import os.path

import dotenv
import hydra
import omegaconf
import pandas as pd
import wandb
from hydra.utils import instantiate
from tqdm import tqdm

dotenv.load_dotenv()

from code_editing.data_sources.extract_code_base import CodeBaseExtractor
from code_editing.configs.evaluation_config import RunEvaluationConfig
from code_editing.metrics.base_metric import BaseMetric


@hydra.main(version_base=None, config_path="conf", config_name="evaluation")
def main(cfg: RunEvaluationConfig):
    """This script evaluates a csv file with columns diff_true and diff_pred."""
    # Read the input file specified by the user
    if cfg.input_path.endswith(".csv"):
        df = pd.read_csv(cfg.input_path)
    else:
        df = pd.read_json(cfg.input_path, lines=True)

    # Get the 'diff_pred' column from the dataframe, replace any NaN values with an empty string
    diff_pred = df["diff_pred"].fillna("")

    # Get the 'diff_true' column from the data source
    diff_true = df["diff_true"]

    # Instantiate the extractor and data source
    extractor: CodeBaseExtractor = instantiate(cfg.extractor)
    data_source = instantiate(cfg.data_source, extractor=extractor)

    res = {}
    pbar = tqdm(cfg.metrics.items(), position=0, desc="Running metrics")

    use_wandb = cfg.wandb.enable
    if use_wandb:
        config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(project=cfg.wandb.project, job_type="evaluation", config=config)
        name = os.path.split(cfg.input_path)[-2]
        wandb.run.name = f"Eval {name}"

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

    res_json = json.dumps(res, indent=2)
    print(res_json)
    # Save the results to the output path
    hydra_output_path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "eval.json")
    with open(hydra_output_path, "w") as f:
        f.write(res_json)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
