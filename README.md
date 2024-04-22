# Code Editing

This project contains a set of code editing tools implemented in Python for the Long Code Arena (LCA) project.

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

## Configuration

The Hydra configuration files for inference and evaluation are located in `scripts/conf`.
These files specify the backbone model, preprocessor, and data source to use, as well as other parameters.

## Available Options
### Backbones
Currently, the following backbones are available:
- `dummy`: A dummy model that randomly picks lines to diff from the code base
- `openai`: The OpenAI API models
- HuggingFace: Various HuggingFace models, including:
  - `llama7`, `llama13` and `llama34`: The [CodeLlama Instruct models](codellama/CodeLlama-7b-Instruct-hf) with 7B, 13B and 34B parameters, respectively
  - `mistral7`: [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
  - `mixtral8x7`: [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
  - Other models can be added by configuring the `backbone` parameter in the hydra configuration

### Data Sources
Currently, the following data sources are available:
- [`lca_code_editing`](https://huggingface.co/datasets/JetBrains-Research/lca-code-editing): The Long Code Arena code editing dataset
- [`ci_fix_python`](https://huggingface.co/datasets/JetBrains-Research/CI-fix-Python): The CI Fix Python dataset
- [`lca_bug_localization`](https://huggingface.co/datasets/JetBrains-Research/lca-bug-localization): The Long Code Arena bug localization dataset

## Evaluation Metrics
The following evaluation metrics are available:
- `chrf`: Character F-score (both for git diff and the predicted code)
- `code_bert_score`: BERTScore F1 (both for git diff and the predicted code)
- `gpt4`: GPT-4 evaluation metric (only git diff)
- `exact_match`: Exact match
- `pass` (Only for the `ci_fix_python` data source): Whether the predicted diff passes the CI (`pass@1`)
- Utility metrics: `is_not_nan` (whether the predicted diff is not NaN), `weak_format` (if the predicted diff is in the correct format), `valid_git_diff` (if the predicted diff is a valid git diff)
