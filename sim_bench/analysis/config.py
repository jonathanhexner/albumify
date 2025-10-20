from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_GLOBAL_CONFIG: Optional["GlobalAnalysisConfig"] = None


@dataclass
class GlobalAnalysisConfig:
    """
    Global settings for analysis utilities.

    - experiment_dir: concrete experiment directory to analyze (default context)
    - base_dir: optional root directory containing experiments (used if experiment_dir not specified)
    - save_dir: directory to save generated figures/reports (created if needed)
    - image_glob: glob for discovering images when paths are missing
    """
    experiment_dir: Optional[Path] = None
    base_dir: Optional[Path] = None
    save_dir: Optional[Path] = None
    image_glob: str = "**/*.{jpg,jpeg,png}"

    def resolve(self) -> "GlobalAnalysisConfig":
        if self.experiment_dir is not None:
            self.experiment_dir = Path(self.experiment_dir)
        if self.base_dir is not None:
            self.base_dir = Path(self.base_dir)
        # Default save_dir under experiment_dir if provided, else under base_dir
        if self.save_dir is None:
            root = self.experiment_dir or self.base_dir or Path(".")
            self.save_dir = Path(root) / "analysis_outputs"
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        return self


def set_global_config(cfg: "GlobalAnalysisConfig") -> "GlobalAnalysisConfig":
    """Set the process-wide analysis configuration (singleton). Returns the resolved config."""
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = cfg.resolve()
    return _GLOBAL_CONFIG


def get_global_config() -> "GlobalAnalysisConfig":
    """Get the process-wide analysis configuration. Raises if not set."""
    if _GLOBAL_CONFIG is None:
        raise ValueError("GlobalAnalysisConfig has not been set. Call set_global_config(...) first.")
    return _GLOBAL_CONFIG


@dataclass
class PlotGridConfig:
    """
    Configuration for grid plotting of query + top-N results.

    - top_n: number of result images to display
    - max_per_row: maximum columns per row for result grid
    - figsize_per_cell: approximate (width, height) inches per cell
    - title_fontsize / label_fontsize: sizes for text
    - cmap: optional colormap for any overlays (unused by default)
    - save: save the figure to disk
    """
    top_n: int = 4
    max_per_row: int = 4
    figsize_per_cell_w: float = 3.0
    figsize_per_cell_h: float = 3.0
    title_fontsize: int = 12
    label_fontsize: int = 10
    cmap: Optional[str] = None
    save: bool = False
    save_filename_suffix: str = "_topn"


