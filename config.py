from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RunConfig:
    model_name: str = 'poisson_pipeline'
    # Guiding text prompt
    sd_2_1: bool = False
    # Which token indices to alter with attend-and-excite
    token_indices: Optional[List[int]] = None
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [42])
    # Path to save all outputs to
    output_path: Path = Path('./output')
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Number of denoising steps to apply attend-and-excite
    max_iter_to_alter: int = 25
    # Resolution of UNet to compute attention maps over
    attention_res: int = 16
    # Whether to run standard SD or attend-and-excite
    run_standard_sd: bool = False
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(default_factory=lambda: {0: 0.9, 5: 0.95, 10: 0.99, 15: 0.995, 20: 0.999})
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 20
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.5))
    # Whether to apply the Gaussian smoothing before computing the maximum attention value for each subject token
    smooth_attentions: bool = True
    # Standard deviation for the Gaussian smoothing
    sigma: float = 0.5
    # Kernel size for the Gaussian smoothing
    kernel_size: int = 3
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = True
    # Whether to save denoising-step attention snapshots
    save_attn_snapshots: bool = True
    # Denoising steps to snapshot, e.g. [0, 10, 17, 25, 35]
    attn_snapshot_steps: List[int] = field(default_factory=lambda: [0, 10, 17, 25])
    # Optional base directory for attention snapshots; if None, uses output_path/attn_progress
    attn_snapshot_base_dir: Optional[Path] = Path('../output/attn_progress')
    # Optional token indices for visualization; defaults to altered token indices when None
    attn_snapshot_token_indices: Optional[List[int]] = None
    # Show attention grids inline when running in notebook contexts
    display_attention_maps: bool = False

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)
