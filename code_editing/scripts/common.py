import concurrent
import json
import logging
import os

import coolname
import hydra
import omegaconf
import pandas as pd
import wandb
from tqdm import tqdm

from code_editing.baseline.baseline import CodeEditor
from code_editing.configs.inference_config import InferenceConfig
from code_editing.data_sources.base_source import CEDataSource
from code_editing.data_sources.hf_source import HuggingFaceSimpleGitCEDataSource
from code_editing.utils import wandb_utils


def get_cool_name():
    return coolname.generate_slug(3)


def inference_loop(
    code_editor: CodeEditor,
    data_source: CEDataSource,
    output_path: str,
    inference_config: InferenceConfig,
):
    """
    Perform inference loop over the dataset for code editing.
    """
    if not isinstance(data_source, HuggingFaceSimpleGitCEDataSource):
        raise ValueError("This script only supports HuggingFaceSimpleGitCEDataSource")

    # Initialize lists to store the true diffs, predicted diffs
    df = pd.DataFrame(columns=["diff_pred", "diff_true", "repo", "base_hash", "message", "viewed_lines"])

    start = inference_config.start_from
    end = inference_config.end_at or len(data_source)
    datapoints = {}
    tries = {i: 0 for i in range(start, end)}

    progress_bar = tqdm(total=end - start, desc="Inference Loop")
    num_added = 0

    def process_datapoint(i):
        X, raw_data = data_source[start + i]
        datapoints[i] = X, raw_data
        return code_editor.generate_diff(X)

    with concurrent.futures.ThreadPoolExecutor(max_workers=inference_config.num_workers) as executor:
        queue = [(executor.submit(process_datapoint, i), i) for i in range(end - start)]
        while queue:
            task_index = {task: i for (task, i) in queue}
            queue = []
            logging.info(
                f"Waiting for {len(task_index)} tasks to complete using {inference_config.num_workers} workers..."
            )
            for task in concurrent.futures.as_completed(task_index):
                i = task_index[task]
                # Get the result
                y_pred = None
                res = {}
                try:
                    res = task.result()
                    y_pred = res["prediction"] + "\n"
                    if y_pred.strip() == "":
                        raise ValueError("Empty prediction")
                except Exception as e:
                    logging.warning(f"Error in inference for #{start + i}", exc_info=e)
                    tries[i] += 1
                    if tries[i] < inference_config.num_tries:
                        queue.append((executor.submit(process_datapoint, i), i))
                        continue
                    else:
                        logging.error(f"Failed to predict for #{start + i}")

                _, raw_data = datapoints[i]
                # Store the results
                viewed_lines = res.get("viewed_lines", {})
                viewed_lines = json.dumps({k: list(v) for k, v in viewed_lines.items() if v})
                try:
                    data = data_source.row_to_data(raw_data)
                    y_true = data_source.row_to_diff(raw_data)
                    df.loc[i] = [y_pred, y_true, data.repo, data.base_hash, data.message, viewed_lines]
                except Exception as e:
                    logging.warning(f"Error in saving the results for #{i}", exc_info=e)
                # Update the progress bar
                progress_bar.update(1)
                num_added += 1
                progress_bar.set_postfix_str(f"latest: {i} {data.repo}@{data.base_hash[:8]}")

                # Save the checkpoint
                if (
                    inference_config.checkpoint_iters is not None
                    and num_added
                    and num_added % inference_config.checkpoint_iters == 0
                ):
                    hydra_output_path = os.path.join(
                        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, os.path.basename(output_path) + f".checkpoint_{num_added}"
                    )
                    logging.info(f"Saving the checkpoint to {hydra_output_path}")
                    my_save_jsonl(df, hydra_output_path)

    # Save the results to wandb
    if wandb.run is not None:
        wandb.log({"prediction": wandb.Table(dataframe=df)})

    # Return the prepared dataframe
    my_save_jsonl(df, output_path)
    # Save to hydra output path
    hydra_output_path = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, os.path.basename(output_path)
    )
    my_save_jsonl(df, hydra_output_path)

    logging.info(f"Saved the results to {output_path}")
    return df


def init_wandb(cfg, run_name, tags=None):
    use_wandb = cfg.inference.wandb.enable
    if use_wandb:
        config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(project=cfg.inference.wandb.project, job_type="inference", config=config, tags=tags)
        wandb.run.name = run_name


def finish_wandb():
    if wandb_utils.is_run_active():
        wandb.finish()


def init_output_path(baseline_name, cfg, data_source):
    output_path = cfg.inference.output_path
    if output_path is None:
        output_path = os.path.join(
            "outputs",
            data_source.name,
            baseline_name,
            "inference.jsonl",
        )
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return output_path


def my_save_jsonl(df: pd.DataFrame, path: str):
    # Hack for https://github.com/ultrajson/ultrajson/issues/252#issuecomment-281978941
    with open(path, "w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
