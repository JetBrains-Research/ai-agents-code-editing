import collections
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import coolname
import omegaconf
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from langchain_community.callbacks import get_openai_callback
from tqdm import tqdm

import wandb
from code_editing.code_editor import CodeEditor
from code_editing.configs.inference_config import InferenceConfig
from code_editing.data_sources.base_source import CEDataSource
from code_editing.data_sources.hf_source import HuggingFaceSimpleGitCEDataSource
from code_editing.utils import wandb_utils

logger = logging.getLogger("inference")


def get_cool_name():
    return coolname.generate_slug(3)


def inference_loop(
    code_editor: CodeEditor,
    data_source: CEDataSource,
    output_path: str,
    inference_config: InferenceConfig,
    run_name: str,
):
    unique_hex = uuid.uuid4().hex[:8]
    model_name = run_name.replace(" ", "_").replace("/", "_").lower() + "_" + unique_hex
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    """
    Perform inference loop over the dataset for code editing.
    """
    if not isinstance(data_source, HuggingFaceSimpleGitCEDataSource):
        raise ValueError("This script only supports HuggingFaceSimpleGitCEDataSource")

    # Initialize lists to store the true diffs, predicted diffs
    df = pd.DataFrame(columns=["diff_pred", "diff_true", "repo", "base_hash", "message", "viewed_lines", "model_name"])

    start = inference_config.start_from
    end = inference_config.end_at or len(data_source)
    datapoints = {}
    tries = {i: 0 for i in range(start, end)}

    progress_bar = tqdm(total=end - start, desc="Inference Loop")
    num_added = 0

    run_summary = {}
    openai_stats = collections.defaultdict(float)

    def process_datapoint(i):
        repo_lock = data_source.get_lock(i)
        with repo_lock, get_openai_callback() as cb:
            datapoints[i] = data_source[i]
            inp = data_source.data_to_input(datapoints[i])
            instance_id = f"{data_source.name}_{i}"
            inp["instance_id"] = instance_id
            try:
                return code_editor.generate_diff(inp)
            finally:
                openai_stats["total_tokens"] += cb.total_tokens
                openai_stats["total_cost"] += cb.total_cost
                openai_stats["successful_requests"] += cb.successful_requests
                openai_stats["completion_tokens"] += cb.completion_tokens
                openai_stats["projected_cost"] = openai_stats["total_cost"] * (end - start) / (num_added + 1)
                wandb.log({"openai": openai_stats})

    with ThreadPoolExecutor(max_workers=inference_config.num_workers) as executor:
        queue = [(executor.submit(process_datapoint, i), i) for i in range(start, end)]
        while queue:
            task_index = {task: i for (task, i) in queue}
            queue = []
            logger.info(
                f"Waiting for {end - start - num_added} tasks to complete using {inference_config.num_workers} workers..."
            )
            for task in as_completed(task_index):
                i = task_index[task]
                # Get the result
                y_pred = None
                res = {}
                data = datapoints.get(i, None)
                try:
                    if data is None:
                        raise ValueError(f"Data for #{i} is None")
                    row_info = f"{data.repo}@{data.base_hash[:8]}"
                    res = task.result()
                    y_pred = res["prediction"] + "\n"
                    if y_pred.strip() == "" and inference_config.skip_empty_diffs:
                        raise ValueError("Empty prediction")
                except Exception as e:
                    if "Empty prediction" in str(e):
                        logger.warning(f"Empty prediction for #{i} {row_info}")
                    else:
                        logger.warning(f"Error in inference for ${i} {row_info}", exc_info=e)
                    tries[i] += 1
                    if tries[i] < inference_config.num_tries:
                        queue.append((executor.submit(process_datapoint, i), i))
                        continue
                    else:
                        logger.error(f"Failed to predict for #{i} {row_info} after {inference_config.num_tries} tries")

                # Get the lines for editing
                viewed_lines = res.get("viewed_lines", {})
                viewed_lines = json.dumps({k: list(v) for k, v in viewed_lines.items() if v})
                # Get run summary
                new_run_summary = res.get("run", {})
                if wandb.run is not None:
                    # upd tools
                    if "tools" in new_run_summary:
                        for tool in new_run_summary["tools"]:
                            for k, v in new_run_summary["tools"][tool].items():
                                run_summary.setdefault("tools", {}).setdefault(tool, {}).setdefault(k, 0)
                                run_summary["tools"][tool][k] += v
                        new_run_summary.pop("tools")
                    # upd rest
                    run_summary.update(new_run_summary)
                    # log
                    wandb.log(run_summary)
                # Add the result to the dataframe
                df.loc[i - start] = [
                    y_pred,
                    data.diff_true,
                    data.repo,
                    data.base_hash,
                    data.message,
                    viewed_lines,
                    model_name,
                ]
                # Update the progress bar
                num_added += 1
                progress_bar.update(1)
                progress_bar.set_postfix_str(f"latest: {i} {row_info}")

                # Save the checkpoint
                if (
                    inference_config.checkpoint_iters is not None
                    and num_added
                    and num_added % inference_config.checkpoint_iters == 0
                ):
                    hydra_output_path = os.path.join(
                        hydra_output_dir,
                        os.path.basename(output_path) + f".checkpoint_{num_added}.jsonl",
                    )
                    logger.info(f"Saving the checkpoint to {hydra_output_path}")
                    my_save_jsonl(df, hydra_output_path)

    # Save the results to wandb
    if wandb.run is not None:
        wandb.log({"prediction": wandb.Table(dataframe=df)})
        wandb.run.tags = wandb.run.tags + (f"unique:{unique_hex}",)

    # Return the prepared dataframe
    my_save_jsonl(df, output_path)
    # Save to hydra output path
    hydra_output_path = os.path.join(hydra_output_dir, os.path.basename(output_path))
    my_save_jsonl(df, hydra_output_path)

    # SWEBench specific
    if hasattr(data_source, "to_swebench_results"):
        swebench_results, instance_ids = data_source.to_swebench_results(df, model_name)
        # save predictions in SWEBench format
        swebench_output_path = os.path.join(hydra_output_dir, f"swebench_preds_{model_name}.json")
        with open(swebench_output_path, "w") as f:
            json.dump(swebench_results, f)
        # save instance ids
        instance_ids_path = os.path.join(hydra_output_dir, f"swe_tasks.txt")
        with open(instance_ids_path, "w") as f:
            f.write("\n".join(instance_ids))
        # log the paths
        logger.info(f"Saved the predictions in SWEBench format to {swebench_output_path}")

    if wandb.run is not None:
        wandb.log_artifact(hydra_output_dir, name=f"inference_results.{model_name}", type="inference_results")
    logger.info(f"Saved the results to {hydra_output_path}")
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
