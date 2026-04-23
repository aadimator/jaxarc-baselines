"""CLI entrypoint for the TTRL benchmark."""

from __future__ import annotations

import hydra
from colorama import Fore, Style
from omegaconf import DictConfig, OmegaConf

from jaxarc_baselines.ttrl import run_ttrl_benchmark


@hydra.main(
    config_path="../experiments/configs",
    config_name="ttrl_benchmark.yaml",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    """Run the TTRL benchmark with reproducible tracking artifacts."""
    OmegaConf.set_struct(cfg, False)
    summary = run_ttrl_benchmark(cfg)
    print(
        f"{Fore.CYAN}{Style.BRIGHT}TTRL benchmark completed. "
        f"Artifacts written to {summary['run_dir']}.{Style.RESET_ALL}"
    )


if __name__ == "__main__":
    main()