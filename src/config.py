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


class GQAConfig(BaseModel):
    """Grouped Query Attention configuration."""
    enabled: bool = True  # If False, use standard MHA (n_kv_heads = n_query_heads)
    n_kv_heads: Optional[int] = None  # If None, defaults to num_heads // 4


class ModelConfig(BaseModel):
    dim: int = 256
    depth: int = 16
    num_heads: int = 8
    topo_dim: int = 3
    mlp_depth: int = 1
    vocab_size: int = 151936 #qwen 3 sized
    global_layer_interval: int = 4
    num_experts: int = 8
    num_active: int = 3
    rope_base: int = 500
    mlp_ratio: float = 4.0
    jitter_noise: float = 0.1
    window_size: float = 10.0
    patch_embedder: PatchEmbedderConfig = Field(default_factory=PatchEmbedderConfig)
    gqa: GQAConfig = Field(default_factory=GQAConfig)

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    @property
    def effective_kv_heads(self) -> int:
        """Get effective number of KV heads for GQA."""
        if not self.gqa.enabled:
            return self.num_heads
        if self.gqa.n_kv_heads is not None:
            return self.gqa.n_kv_heads
        return max(1, self.num_heads // 4)


# =============================================================================
# Dataset Components
# =============================================================================

class NoiseParams(BaseModel):
    min_snr: float = -4.0
    max_snr: float = 2.0
    angle_range_deg: float = 30.0
    jitter_pct: float = 0.05


class TimeSamplerConfig(BaseModel):
    min_pct: float = 0.01
    max_pct: float = 1.0
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
    type: Literal["checkerboard", "torus", "video", "fractal", "sprite_atlas", "procedural"]
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

class MuonConfig(BaseModel):
    """Muon optimizer config for transformer layers."""
    lr: float = 0.02
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    state_dtype: str = "bf16"  # "fp32" or "bf16" - bf16 is safe (no v accumulator)


class AdamWConfig(BaseModel):
    """AdamW optimizer config for embedding/norm layers."""
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)


class SchedulerConfig(BaseModel):
    """Learning rate scheduler config."""
    type: str = "onecycle"  # "onecycle", "cosine", "none"
    pct_start: float = 0.1
    div_factor: float = 10.0
    final_div_factor: float = 25.0
    min_lr: float = 0.0  # For cosine


class OptimizerConfig(BaseModel):
    """Optimizer configuration - supports both simple AdamW and heterogeneous Muon+AdamW.

    For simple AdamW (backwards compatible):
        type = "adamw" (default)
        lr, weight_decay, etc. at top level

    For heterogeneous Muon+AdamW:
        type = "heterogeneous"
        muon = {...} for transformer layers
        adamw = {...} for embedding/norm layers
        scheduler = {...} for all groups
    """
    # Common fields (backwards compatible with simple AdamW)
    type: str = "adamw"  # "adamw" or "heterogeneous"
    lr: float = 5e-4
    weight_decay: float = 0.1
    max_lr: float = 5e-4
    pct_start: float = 0.1
    div_factor: float = 10.0
    final_div_factor: float = 100.0
    betas: Tuple[float, float] = (0.9, 0.95)

    # Heterogeneous optimizer sub-configs
    muon: Optional[MuonConfig] = None
    adamw: Optional[AdamWConfig] = None
    scheduler: Optional[SchedulerConfig] = None
    targeting: Optional["ParameterTargetingConfig"] = None  # Forward ref, resolved below

    @model_validator(mode="after")
    def validate_heterogeneous_requires_subconfigs(self):
        """Ensure muon and adamw subconfigs are present when type == 'heterogeneous'."""
        if self.type == "heterogeneous":
            if self.muon is None:
                raise ValueError("type='heterogeneous' requires [training.optimizer.muon] config")
            if self.adamw is None:
                raise ValueError("type='heterogeneous' requires [training.optimizer.adamw] config")
        return self


class AEOptimizerConfig(BaseModel):
    lr: float = 1e-3
    weight_decay: float = 0.01
    max_lr: float = 1e-3
    pct_start: float = 0.1


class ParameterTargetingConfig(BaseModel):
    """Configuration for parameter classification in heterogeneous optimizer.

    Controls which parameters go to Muon vs AdamW based on name patterns.
    Patterns are matched case-insensitively against parameter names.

    Priority order: embedding > norm > ae > fsq_adjacent > transformer (Muon)
    """
    # Patterns for embedding/unembedding layers -> AdamW
    embedding_patterns: List[str] = Field(
        default_factory=lambda: ['embed', 'head', 'lm_head', 'wte', 'wpe', 'embedder', 'unembedder']
    )
    # Patterns for norm layers -> AdamW (no weight decay)
    norm_patterns: List[str] = Field(
        default_factory=lambda: ['norm', 'ln', 'layernorm', 'rmsnorm']
    )
    # Patterns for AE components -> AdamW (heterogeneous gradients from FSQ/sparsity)
    ae_patterns: List[str] = Field(
        default_factory=lambda: ['sparse_ae', 'encoders', 'decoders', 'level_logsnr']
    )
    # Patterns for FSQ-adjacent params -> AdamW (sigmoid STE attenuates gradients)
    # Note: code_proj inside encoders still matches, latent_code_proj/unproj are the wrapper projections
    fsq_patterns: List[str] = Field(
        default_factory=lambda: ['code_proj', 'latent_code_proj', 'latent_code_unproj', 'fsq', 'sparsity', 'dim_logits', 'level_values', 'attn_gate', 'logsnr']
    )


# Resolve forward reference in OptimizerConfig for ParameterTargetingConfig
OptimizerConfig.model_rebuild()


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
    caching_resolution: int = 128
    image_buckets: List[ResolutionBucketConfig] = Field(default_factory=list)
    video_buckets: List[VideoBucketConfig] = Field(default_factory=list)

class OnlineVarianceCorrectionConfig(BaseModel):
    enabled: bool = False
    num_buckets: int = 20
    snr_min: float = -4.0
    snr_max: float = 6.0
    ema_decay: float = 0.99
    warmup_steps: int = 100  # Steps before correction kicks in


class AEAttentionConfig(BaseModel):
    """Attention configuration for sparse AE transformer layers."""
    mode: Literal["full", "sliding", "gemma", "gemma_bigbird"] = "gemma"  # gemma = 3 local + 1 global
    window_size: int = 4  # For sliding window: attend to ±window_size patches
    global_layer_interval: int = 4  # For gemma mode: every Nth layer is global
    n_query_heads: int = 8
    n_kv_heads: int = 2  # GQA ratio
    n_global_tokens: int = 4    # For bigbird/gemma_bigbird: number of register tokens
    bigbird_layout: list = [2, 2]  # For gemma_bigbird: [n_local, n_bigbird] layers per cycle
    random_min_k: int = 3       # Random attention: at least this many random keys
    random_min_p: float = 0.02    # Random attention: at least this % of seq_len (whichever is larger)


class TopologyGeometryConfig(BaseModel):
    """Configuration for topology coordinate computation and distance metrics.

    Controls how tokens are positioned in the R^n topology space that RnRoPE
    and SWA masks use for distance computations. Different geometries enable
    different diffusion modes (pixel-space vs latent-space) via config alone.

    The topology has dimensions: [highway, spatial_1, spatial_2, ..., level?]
    - highway: monotonic context position (always present)
    - spatial_*: grid coordinates for images, origin for text
    - level: residual quantization level (only for latent diffusion)
    """
    # Diffusion space: "pixel" = patch embeddings, "latent" = AE code embeddings
    diffusion_space: Literal["pixel", "latent"] = "pixel"

    # Whether to include level as a topology dimension (auto-enabled for latent)
    include_level_dim: bool = False

    # Distance metric for level dimension in SWA masks
    # dist² = spatial_dist² + (level_lambda * level_dist)²
    level_lambda: float = 0.5  # Cost of crossing levels relative to spatial

    # Level coordinate scaling (maps level index to coordinate value)
    # Larger values = more separation between levels in RoPE frequencies
    level_scale: float = 1.0

    # Whether same-position cross-level attention is always allowed (ignores SWA window)
    # Creates "vertical tubes" through the level stack
    vertical_attention_free: bool = True

    # Optional: custom distance metric for advanced use cases
    # "euclidean" = standard sqrt(sum of squares)
    # "product_geodesic" = treats level crossings as graph edges
    distance_metric: Literal["euclidean", "product_geodesic"] = "euclidean"


class AELossScheduleConfig(BaseModel):
    """Loss schedule for AE training - lerp from MSE to BCE over training.

    Early training uses MSE for smooth gradients and coarse structure learning.
    Late training shifts toward BCE to push for sharp, committed predictions.
    """
    enabled: bool = False
    # Start weights (beginning of training)
    mse_start: float = 1.0
    bce_start: float = 0.0
    # End weights (end of training)
    mse_end: float = 0.1
    bce_end: float = 0.9
    # Schedule shape: "linear", "cosine", or "step" (switch at pct_switch)
    schedule: Literal["linear", "cosine", "step"] = "linear"
    pct_switch: float = 0.8  # For "step" schedule: switch at this fraction of training


class DiffusionLossScheduleConfig(BaseModel):
    """Loss schedule for diffusion (v-field) training - lerp from MSE to partial BCE.

    Tests whether BCE gradients on sigmoid(v) find better v-fields than MSE.
    Mild intervention: start with 100% MSE, lerp to ~90% MSE / 10% BCE.

    For v-field targets: applies sigmoid(v_pred) and sigmoid(v_target) before BCE,
    treating the velocity field as logits.
    """
    enabled: bool = False
    # Start weights (beginning of diffusion training)
    mse_start: float = 1.0
    bce_start: float = 0.0
    # End weights (end of diffusion training)
    mse_end: float = 0.9
    bce_end: float = 0.1
    # Schedule shape: "linear", "cosine", or "step"
    schedule: Literal["linear", "cosine", "step"] = "linear"
    pct_switch: float = 0.8  # For "step" schedule


class SparseAEMoEConfig(BaseModel):
    """MoE configuration for weight-shared swiglu_moe variant."""
    num_experts: int = 16  # Total experts in each MoE layer
    num_active: int = 3    # Active experts per token (routes to top-k)
    jitter_noise: float = 0.1  # Router noise during training (load balancing)


class SparseAEConfig(BaseModel):
    """Configuration for kmaze_ae sparse hierarchical autoencoder."""
    enabled: bool = False
    ae_type: Literal["sparse_dim", "swiglu", "swiglu_moe"] = "sparse_dim"  # sparse_dim=3-bit, swiglu=binary+2D RoPE, swiglu_moe=weight-shared MoE
    # MoE configuration (only used when ae_type='swiglu_moe')
    moe: SparseAEMoEConfig = Field(default_factory=SparseAEMoEConfig)
    n_levels: int = 6
    patch_size: int = 16
    hidden_dim: int = 256
    code_dim: int = 128
    # Sparsity mode: per_level = same dims active for all patches (learned), per_patch = content-dependent
    # per_level is more stable, per_patch is more expressive but prone to collapse
    sparsity_mode: Literal["per_level", "per_patch"] = "per_level"
    # Wavelet subspace routing: partitions code_dim into wavelet and amplitude subspaces
    # Sparsity pattern across subspaces encodes pathway selection (no sigmoid gating needed)
    wavelet_gating: bool = False  # Enable subspace-routed encoder/decoder
    n_wavelet_dims: Optional[int] = None  # Wavelet subspace size (default: code_dim // 2)
    # Entropy regularization to prevent subspace collapse (all codes routing to one subspace)
    routing_entropy_weight: float = 0.0  # Weight for entropy regularization (0 = disabled)
    k_per_patch: int = 4  # Sparsity control: final k (keep k of code_dim dims)
    # K-annealing: exponential decay from k_start to k_per_patch over k_anneal_steps
    # Curriculum: start with more active dims (easier task), progressively constrain
    k_start: Optional[int] = None  # Starting k (if None, no annealing - use k_per_patch)
    k_anneal_steps: int = 2000  # Steps over which to anneal k
    residual_scale: float = 2.0
    fourier_dim: int = 16
    ae_loss_weight: float = 0.1  # Weight for diffusion-predicted reconstruction in joint training
    direct_ae_weight: float = 0.1  # Weight for DIRECT AE reconstruction (encoder→decoder, no diffusion)
    logsnr_loss_weight: float = 0.1  # Weight for logsnr prediction in joint training
    # Loss function type for AE training
    # cumulative_mse: average MSE across all level reconstructions (reference impl)
    # final_mse: MSE only on final reconstruction
    # cumulative_mse_contrib: MSE + penalty for levels that contribute too little
    loss_type: Literal["cumulative_mse", "final_mse", "cumulative_mse_contrib"] = "cumulative_mse"
    # Loss schedule: lerp from MSE to BCE over training for sharper reconstructions
    loss_schedule: AELossScheduleConfig = Field(default_factory=AELossScheduleConfig)
    # Diffusion (v-field) loss schedule: lerp from MSE to partial BCE during diffusion training
    diffusion_loss_schedule: DiffusionLossScheduleConfig = Field(default_factory=DiffusionLossScheduleConfig)
    n_layers: int = 4  # Transformer layers per encoder/decoder
    attention: AEAttentionConfig = Field(default_factory=AEAttentionConfig)
    # Latent diffusion training mode (uses [training].steps for step count)
    latent_diffusion: bool = False
    # Topology geometry for latent diffusion (controls R^n embedding + distance metrics)
    topology: TopologyGeometryConfig = Field(default_factory=TopologyGeometryConfig)


class GraphCaptureConfig(BaseModel):
    """CUDA Graph capture configuration for model forward pass.

    Graph capture requires warmup with REAL data before capture.
    One graph is captured per sequence length bucket.
    """
    enabled: bool = False
    warmup_steps: int = 3  # Number of warmup iterations before capture (minimum 3)
    capture_after_warmup: bool = True  # Auto-capture after warmup completes
    use_dedicated_stream: bool = True  # Use separate CUDA stream for capture


class PrecisionConfig(BaseModel):
    """Mixed precision configuration for FP8 weight storage.

    FP8 stores transformer weights in 8-bit (half memory), computes in bf16.
    Embeddings and norms stay in bf16 for numerical stability.
    """
    weights: str = "bf16"  # "fp8" for 8-bit weights, "bf16" for default
    activations: str = "bf16"  # activations always bf16 for now
    skip_patterns: List[str] = Field(
        default_factory=lambda: ["embed", "head", "norm", "embedder", "unembedder"]
    )


class FractalParams(BaseModel):
    """Parameters for procedural fractal generation."""
    seed: int = 42
    fractal_types: List[str] = Field(default_factory=lambda: ["mandelbrot", "julia", "burning_ship"])
    color_palette: str = "random"  # "random", "fire", "ice", "earth"
    max_iterations: int = 256
    zoom_range: Tuple[float, float] = (0.5, 4.0)
    resolution: Optional[int] = None
    text_position: str = "none"


class ProceduralParams(BaseModel):
    """Parameters for procedural noise/pattern generation.

    Generates layered compositions with base generators + effect layers blended
    using photoshop-style blend modes. All generation is batched on GPU.
    """
    seed: int = 42
    # Layer composition settings
    min_layers: int = 1  # Minimum effect layers on top of base
    max_layers: int = 10  # Maximum effect layers
    # Generator selection (None = use all available)
    base_types: Optional[List[str]] = None  # e.g. ["pink_noise", "gradient", "solid"]
    effect_types: Optional[List[str]] = None  # e.g. ["moire", "crosshatch", "shapes"]
    # Available blend modes: overlay, add, subtract, multiply, screen, hard_light, soft_light, difference, exclusion
    blend_modes: Optional[List[str]] = None  # None = use all
    # Distortion
    bulge_probability: float = 0.3  # Probability of applying bulge distortion per layer

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

    # Bucketing Config
    bucketing: BucketingConfig = Field(default_factory=BucketingConfig)
    precision: str = "fp32"  # Options: "fp32", "bf16", "fp16"
    precision_config: PrecisionConfig = Field(default_factory=PrecisionConfig)  # FP8 weight config
    online_variance_correction: OnlineVarianceCorrectionConfig = Field(default_factory=OnlineVarianceCorrectionConfig)
    sparse_ae: SparseAEConfig = Field(default_factory=SparseAEConfig)
    graph_capture: GraphCaptureConfig = Field(default_factory=GraphCaptureConfig)


class SubspaceSensitivityConfig(BaseModel):
    """Configuration for wavelet/amplitude subspace sensitivity sweep."""
    enabled: bool = False  # Run sensitivity sweep after AE training
    ablation_rates: List[float] = Field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    n_trials: int = 5  # Stochastic trials per ablation rate (averaged)
    n_samples: int = 16  # Number of images to evaluate
    resolutions: List[int] = Field(default_factory=lambda: [64, 128])  # Resolutions to test


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
    # Custom eval queries (loaded from eval_configs)
    queries: List[Dict[str, Any]] = Field(default_factory=list)
    # Subspace sensitivity sweep config (for wavelet-gating FSQ AE)
    subspace_sensitivity: SubspaceSensitivityConfig = Field(default_factory=SubspaceSensitivityConfig)
 

class PageTableConfig(BaseModel):
    num_blocks: int = 1024
    block_size: int = 128
    max_batch_size: int = 128
    max_logical_blocks: int = 1024


class EvalServerConfig(BaseModel):
    """Configuration for eval server integration (network-yeet weights)."""
    enabled: bool = False  # Yeet weights to eval server at end of training
    url: str = "http://localhost:8421"  # Eval server URL
    health_check: bool = True  # Query health after yeet to verify transfer


class LoggingConfig(BaseModel):
    output_dir: Path = Path("./experiments_mix")
    log_interval: int = 100
    sample_after_training: bool = True
    eval_server: EvalServerConfig = Field(default_factory=EvalServerConfig)

# =============================================================================
# Root Config
# =============================================================================

class ExperimentConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    page_table: PageTableConfig = Field(default_factory=PageTableConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    # soon deprecating~!
    dataset_mix: Dict[str, DatasetSplit] = Field(default_factory=dict)
    
    # Modular Config Paths
    dataset_configs: List[str] = Field(default_factory=list)
    eval_configs: List[str] = Field(default_factory=list)

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

def load_data_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomli.load(f)

def load_eval_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomli.load(f)

def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    """
    Loads TOML, processes modular includes, and sanitizes to Dict.
    """
    if path is None:
        base_data = {}
        root_dir = Path(".")
    else:
        p = Path(path)
        root_dir = p.parent
        with open(p, "rb") as f:
            base_data = tomli.load(f)
    
    # 1. Merge Modular Dataset Configs
    ds_files = base_data.get('dataset_configs', [])
    if ds_files:
        if 'dataset_mix' not in base_data:
            base_data['dataset_mix'] = {}
            
        for ds_path in ds_files:
            p = Path(ds_path)
            if not p.is_absolute():
                p = root_dir / p
            
            if p.exists():
                ds_data = load_data_config(p)
                # Merge logic: If 'dataset_mix' is at root, use it; otherwise assume keys are splits
                to_merge = ds_data.get('dataset_mix', ds_data)
                base_data['dataset_mix'].update(to_merge)
            else:
                print(f"⚠️ Warning: Dataset config not found: {p}")

    # 2. Merge Modular Eval Configs
    eval_files = base_data.get('eval_configs', [])
    if eval_files:
        # Ensure path exists structure
        if 'sampling' not in base_data: base_data['sampling'] = {}
        if 'queries' not in base_data['sampling']: base_data['sampling']['queries'] = []
            
        for eval_path in eval_files:
            p = Path(eval_path)
            if not p.is_absolute():
                p = root_dir / p
                
            if p.exists():
                eval_data = load_eval_config(p)
                queries = eval_data.get('queries', [])
                if queries:
                    print(f"  Loaded {len(queries)} eval queries from {p.name}")
                    base_data['sampling']['queries'].extend(queries)
            else:
                print(f"⚠️ Warning: Eval config not found: {p}")
    
    # 3. Validate and Sanitize
    raw_cfg = ExperimentConfig(**base_data)
    return sanitize_config(raw_cfg)

if __name__ == "__main__":
    import json
    cfg = ExperimentConfig()
    print(json.dumps(cfg.to_dict(), indent=2, default=str))
