from dataclasses import dataclass


@dataclass
class AgentExecutorConfig:
    verbose: bool = False
    handle_parsing_errors: bool = True
    return_intermediate_steps: bool = True
