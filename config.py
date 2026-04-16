from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from utils.attention_utils import AttentionConfig


@dataclass
class RunConfig:
    model_name: str = 'poisson_pipeline'
    # Guiding text prompt
    sd_2_1: bool = False
    # Which token indices to alter with attend-and-excite
    token_indices: Optional[List[int]] = None
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [42, 6143, 7792, 8892, 9010])
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
    # Attention visualization and storage configuration
    attention_config: Optional[AttentionConfig] = None

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)
        # Default attention config if not provided
        if self.attention_config is None:
            self.attention_config = AttentionConfig(
                save=True,
                save_steps=[0, 10, 17, 25],
                save_dir=self.output_path / 'attn_progress',
                display=False
            )
