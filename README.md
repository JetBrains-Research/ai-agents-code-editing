# Code Editing

This project contains a set of code editing tools implemented in Python.

## Usage

### Inference
This project contains a set of scripts for running inference using different interaction strategies and backbones.
The scripts use Hydra for configuration. They produce a JSONL file containing the predicted diffs.
#### Baselines

To run a baseline, use the `run_baseline.py` script.

Example usage:

```bash
python scripts/run_baseline.py backbone=llama7 data_source=lca_code_editing preprocessor=truncate backbone/prompt=fewshot2
```
#### Agents

To run an agent, use the `run_agent.py` script.
This scripts uses Hydra for configuration.

Example usage:

```bash
python scripts/run_agent.py backbone=llama7 data_source=lca_code_editing
```

To view all available options, take a look at the `scripts/conf` directory.

### Evaluation

To evaluate the results, use the `run_evaluation.py` script.
The script outputs a JSON containing the evaluation results for various metrics specified in the configuration.

Example usage:

```bash
python scripts/run_evaluation.py input_path=inference.csv
```
