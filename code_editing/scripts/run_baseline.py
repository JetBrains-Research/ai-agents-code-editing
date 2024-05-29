import dotenv
import hydra
from hydra.utils import instantiate

dotenv.load_dotenv()

from code_editing.backbones import CEBackbone
from code_editing.baseline import CEBaseline
from code_editing.configs.baseline_config import RunBaselineConfig
from code_editing.data_sources.base_source import CEDataSource
from code_editing.data_sources.extract_code_base import CodeBaseExtractor
from code_editing.scripts.common import finish_wandb, inference_loop, init_output_path, init_wandb


@hydra.main(version_base=None, config_path="conf", config_name="baseline")
def main(cfg: RunBaselineConfig):
    # Initialize extractor and data source
    extractor: CodeBaseExtractor = instantiate(cfg.extractor)
    data_source: CEDataSource = instantiate(cfg.data_source, extractor=extractor)

    # Instantiate
    preprocessor = instantiate(cfg.preprocessor, model_name=cfg.backbone.model_name)
    backbone: CEBackbone = instantiate(cfg.backbone)
    baseline = CEBaseline(backbone, preprocessor)

    # Output path
    output_path = init_output_path(backbone.name, cfg, data_source)

    # WandB initialization
    run_name = f"Baseline {backbone.name}"
    init_wandb(cfg, run_name)

    # Perform inference loop and save predictions
    inference_loop(
        baseline,
        data_source,
        output_path,
        cfg.inference,
    )
    finish_wandb()


if __name__ == "__main__":
    main()
