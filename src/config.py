# src/config.py - Pydantic configuration schema
"""
Pydantic configuration for bench_multisnr_zc.py (Field Diffusion).
Matches nu-spec.md exactly.
# src/config.py - Pydantic configuration schema
"""
from pathlib import Path
from typing import Literal, Dict, List, Optional, Tuple, Union, Any
from pydantic import BaseModel, Field, model_validator
import tomli

import logging

# =============================================================================
# Model Components
# =============================================================================

class PatchEmbedderConfig(BaseModel):
    input_channels: int = 3
    fourier_dim: int = 16
    context_size: int = 4
    stride: int = 2
    mlp_depth: int = 1


class ModelConfig(BaseModel):
    dim: int = 256
    depth: int = 16
    num_heads: int = 8
    topo_dim: int = 3
    mlp_depth: int = 1
    vocab_size: int = 65536
    global_layer_interval: int = 4
    num_experts: int = 8
    num_active: int = 3
    rope_base: int = 500
    mlp_ratio: float = 4.0
    jitter_noise: float = 0.1
    window_size: float = 10.0
    patch_embedder: PatchEmbedderConfig = Field(default_factory=PatchEmbedderConfig)
    
    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


# =============================================================================
# Dataset Components
# =============================================================================

class NoiseParams(BaseModel):
    min_snr: float = -4.0
    max_snr: float = 2.0
    angle_range_deg: float = 30.0
    jitter_pct: float = 0.05


class TimeSamplerConfig(BaseModel):
    min_pct: float = 0.001
    max_pct: float = 0.05
    stride: Optional[int] = None


class SequenceFrame(BaseModel):
    res: int = 32
    relative_res: float = 1.0 
    noise_mode: Literal["uniform", "split"] = "uniform"
    noise_params: NoiseParams = Field(default_factory=NoiseParams)


class VideoParams(BaseModel):
    path: str
    time_sampler: TimeSamplerConfig = Field(default_factory=TimeSamplerConfig)
    sequence_structure: List[SequenceFrame] = Field(default_factory=list)


class DatasetSplit(BaseModel):
    type: Literal["checkerboard", "torus", "video"]
    ratio: float = 1.0
    noise_mode: Literal["uniform", "split"] = "uniform"
    noise_params: NoiseParams = Field(default_factory=NoiseParams)
    
    # FIX: Enforce non-nullable dictionary. 
    # Missing TOML key -> {} instead of None.
    params: Dict[str, Any] = Field(default_factory=dict)
    
    def to_iterator_config(self) -> Dict[str, Any]:
        """Convert to format expected by CompositeIterator."""
        # Now we can just dump, knowing params is safe
        return {
            "type": self.type,
            "ratio": self.ratio,
            "noise_mode": self.noise_mode,
            "noise_params": self.noise_params.model_dump(),
            "params": self.params
        }


# =============================================================================
# Training Components
# =============================================================================

class OptimizerConfig(BaseModel):
    lr: float = 5e-4
    weight_decay: float = 0.1
    max_lr: float = 5e-4
    pct_start: float = 0.1
    div_factor: float = 10.0
    final_div_factor: float = 100.0


class AEOptimizerConfig(BaseModel):
    lr: float = 1e-3
    weight_decay: float = 0.01
    max_lr: float = 1e-3
    pct_start: float = 0.1

class ResolutionBucketConfig(BaseModel):
    resolution: int
    batch_size: Optional[int] = None
    weight: float = 1.0

class VideoBucketConfig(BaseModel):
    context_resolution: int
    target_resolution: int
    num_context_frames: int = 3
    batch_size: Optional[int] = None

class BucketingConfig(BaseModel):
    enabled: bool = False
    base_resolution: int = 32
    base_batch_size: int = 128
    image_buckets: List[ResolutionBucketConfig] = Field(default_factory=list)
    video_buckets: List[VideoBucketConfig] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    ae_steps: int = 500
    steps: int = 500
    lambda_coeff: float = 0.2
    mode: Literal["naive", "factorized"] = "naive"
    batch_size: int = 8
    compile: bool = True
    compile_dynamic: bool = True
    schedule_bounds: Tuple[float, float] = (5.0, -4.0)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    ae_optimizer: AEOptimizerConfig = Field(default_factory=AEOptimizerConfig)
        
    # NEW: Register Bucketing Config
    bucketing: BucketingConfig = Field(default_factory=BucketingConfig)

    precision: str = "fp32"  # <--- ENABLE THIS # Options: "fp32", "bf16", "fp16"


class SamplingConfig(BaseModel):
    num_samples: int = 8
    steps: int = 50
    target_logsnr: float = 10.0
    resolutions: List[int] = Field(default_factory=lambda: [32, 64])

    # New Causal Sweep fields
    enable_sweep: bool = False
    sweep_count: int = 4
    sweep_length: int = 4
    sweep_range: Tuple[float, float] = (2.0, -4.0)

class PageTableConfig(BaseModel):
    num_blocks: int = 1024
    block_size: int = 128
    max_batch_size: int = 128
    max_logical_blocks: int = 1024


class LoggingConfig(BaseModel):
    output_dir: Path = Path("./experiments_mix")
    log_interval: int = 100
    sample_after_training: bool = True


# =============================================================================
# Root Config
# =============================================================================

class ExperimentConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    page_table: PageTableConfig = Field(default_factory=PageTableConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    dataset_mix: Dict[str, DatasetSplit] = Field(default_factory=dict)
    
    @model_validator(mode="after")
    def validate_model(self):
        if self.model.dim % self.model.num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        return self
    
    def get_dataset_mix_dict(self) -> Dict[str, Any]:
        """Convert dataset_mix to format expected by CompositeIterator."""
        return {k: v.to_iterator_config() for k, v in self.dataset_mix.items()}
    
    @classmethod
    def from_toml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "rb") as f:
            data = tomli.load(f)
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def sanitize_config(cfg: Union[ExperimentConfig, Dict[str, Any]]) -> Dict[str, Any]:
    """
    The Great Filter.
    1. Converts Pydantic 'Magical Dreamland' objects into honest, working-class Dictionaries.
    2. Strips away dot-notation access privileges.
    3. Ensures downstream code crashes on missing keys instead of guessing.
    """
    if isinstance(cfg, BaseModel):
        # model_dump() converts nested Pydantic models to nested dicts automatically
        return cfg.model_dump()
    elif isinstance(cfg, dict):
        return cfg
    else:
        raise TypeError(f"Config must be Pydantic Model or Dict, got {type(cfg)}")

def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """
    Loads TOML and immediately sanitizes it to a Dict.
    Downstream code should never see an ExperimentConfig object.
    """
    if path is None:
        raw_cfg = ExperimentConfig()
    else:
        raw_cfg = ExperimentConfig.from_toml(path)
    
    return sanitize_config(raw_cfg)


if __name__ == "__main__":
    import json
    cfg = ExperimentConfig()
    print(json.dumps(cfg.to_dict(), indent=2, default=str))
