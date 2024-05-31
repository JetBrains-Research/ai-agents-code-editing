# On The Importance of Reasoning for Context Retrieval in Repository-Level Code Editing
This repository contains the code for the paper
"On The Importance of Reasoning for Context Retrieval in Repository-Level Code Editing".

We provide a comprehensive evaluation of the context retrieval task with different reasoning strategies.

This framework can be further extended to evaluate other context retrieval models for code editing.

## How to Use
### Installation
This project uses [`poetry`](https://python-poetry.org/) for dependency management. To install the dependencies, run:

```bash
poetry install
```

To activate the virtual environment, run:

```bash
poetry shell
```

### Quick Start

To run the experiments, use the following command:

```bash
export PYTHONPATH=.
export CE_DATA_PATH=cache_dir

python code_editing/scripts/run_agent.py
```

Our framework uses the `hydra` library for configuration management. You can customize the configuration by modifying the `code_editing/scripts/conf` directory.

To run the experiments with a specific configuration, use the following command:

```bash
python code_editing/scripts/run_agent.py -cn <config_name> tools=<toolkit> data_source=<data_source>
```

Available configuration names are:
- `baseline` (bm25 retrieval with unmodified query)
- `agent` (tool call criterion)
- `agent_cl` (context length criterion)
- `agent_sr` (self reflection criterion)
- `acr` ([AutoCodeRover](https://github.com/nus-apr/auto-code-rover))
- `my_acr` (AutoCodeRover with langchain-based retrieval)

In addition, different toolkits and data sources can be specified, such as:
- `retrieval_search` (langchain-based retrieval, bm25 by default)
- `acr_toolkit` (AST tools from AutoCodeRover)

By default, SWE-Bench_Lite is used as the data source.

For example, to run the experiments with Self-Reflection stopping criterion and AST tools from AutoCodeRover on full SWE-Bench dataset, use the following command:
```bash
python code_editing/scripts/run_agent.py -cn agent_sr tools=acr_toolkit data_source=swe_bench data_source.lite=false
```

More details can be found in the `code_editing/scripts/conf` and `code_editing/configs` directories.

### Evaluation
To evaluate the results, use the following command:

```bash
python code_editing/scripts/evaluate.py input_path=/path/to/inference.jsonl
```

The evaluation script will output the evaluation metrics for the inference results.
To configure the evaluation, modify the `code_editing/scripts/conf/evaluation.yaml` file.

[//]: # (TODO: Citation)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
