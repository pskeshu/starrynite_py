"""Pydantic configuration schemas for the StarryNite pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, computed_field


class ImagingConfig(BaseModel):
    """Microscopy imaging parameters."""

    xy_res: float = Field(description="XY pixel size in microns")
    z_res: float = Field(description="Z step size in microns")
    n_slices: int = Field(description="Number of Z slices per volume")
    n_channels: int = Field(default=1, description="Number of imaging channels")
    nuclear_channel: int = Field(default=0, description="Channel index for nuclear signal")
    time_prefix: str = Field(default="_t", description="Prefix before timepoint number in filenames")
    time_digits: int = Field(default=3, description="Number of digits for timepoint (e.g. 3 for t001)")

    @computed_field
    @property
    def anisotropy(self) -> float:
        """Z/XY resolution ratio for anisotropic processing."""
        return self.z_res / self.xy_res


class StarDistConfig(BaseModel):
    """StarDist 3D detection parameters."""

    model_name: str = Field(default="3D_demo", description="Pretrained model name or path to custom model")
    prob_thresh: float | None = Field(default=None, description="Detection probability threshold (None=auto)")
    nms_thresh: float | None = Field(default=None, description="Non-maximum suppression threshold (None=auto)")
    scale: list[float] | None = Field(default=None, description="Per-axis scaling before prediction [z, y, x]")
    n_tiles: list[int] | None = Field(default=None, description="Tile grid for memory-efficient prediction [z, y, x]")
    normalize: bool = Field(default=True, description="Percentile-normalize input images")
    normalize_low: float = Field(default=1.0, description="Lower percentile for normalization")
    normalize_high: float = Field(default=99.8, description="Upper percentile for normalization")


class UltrackConfig(BaseModel):
    """ultrack tracking parameters."""

    min_area: int = Field(default=100, description="Minimum nucleus area in voxels")
    max_area: int = Field(default=10000, description="Maximum nucleus area in voxels")
    max_distance: float = Field(default=50.0, description="Maximum frame-to-frame linking distance")
    division_weight: float = Field(default=-0.1, description="ILP weight for division events")
    appear_weight: float = Field(default=-0.5, description="ILP penalty for track appearance")
    disappear_weight: float = Field(default=-0.5, description="ILP penalty for track disappearance")


class BtrackConfig(BaseModel):
    """btrack tracking parameters."""

    max_search_radius: float = Field(default=50.0, description="Maximum search radius for linking")
    motion_model: Literal["constant_velocity", "random_walk"] = Field(
        default="constant_velocity", description="Motion model type"
    )
    hypothesis_model: str = Field(default="cell_hypothesis", description="Hypothesis model config name")
    enable_divisions: bool = Field(default=True, description="Enable cell division detection")
    optimization_method: Literal["EXACT", "APPROXIMATE"] = Field(
        default="EXACT", description="EXACT for <1000 cells, APPROXIMATE for >1000"
    )
    max_lost: int = Field(default=5, description="Max frames a track can be lost before termination")


class DataConfig(BaseModel):
    """Data paths and naming configuration."""

    input_dir: Path = Field(description="Directory containing TIFF stacks")
    embryo_name: str = Field(description="Embryo name prefix for file matching")
    output_dir: Path = Field(default=Path("output"), description="Directory for pipeline output")
    ground_truth_dir: Path | None = Field(default=None, description="Directory with GT nuclei files for evaluation")
    start_time: int = Field(default=0, description="First timepoint to process")
    end_time: int | None = Field(default=None, description="Last timepoint (None=all available)")


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    imaging: ImagingConfig
    data: DataConfig
    stardist: StarDistConfig = Field(default_factory=StarDistConfig)
    ultrack: UltrackConfig = Field(default_factory=UltrackConfig)
    btrack: BtrackConfig = Field(default_factory=BtrackConfig)
    tracker: Literal["ultrack", "btrack"] = Field(default="ultrack", description="Which tracker to use")
