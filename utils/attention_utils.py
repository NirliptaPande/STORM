"""Utilities for handling attention visualization and storage."""

from pathlib import Path
from typing import List, Optional
from utils.vis_utils import visualize_cross_attention_maps
from utils.ptp_utils import AttentionStore


class AttentionConfig:
    """Configuration for attention visualization and storage."""
    
    def __init__(
        self,
        save: bool = False,
        save_steps: Optional[List[int]] = None,
        save_dir: Optional[Path] = None,
        token_indices: Optional[List[int]] = None,
        display: bool = False,
    ):
        """
        Args:
            save: Whether to save attention maps at specific steps
            save_steps: Which diffusion steps to save attention maps at (e.g., [0, 10, 20])
            save_dir: Base directory to save attention maps to
            token_indices: Which token indices to visualize; if None, uses all altered tokens
            display: Whether to display attention maps inline
        """
        self.save = save
        self.save_steps = set(save_steps or [])
        self.save_dir = Path(save_dir) if save_dir else None
        self.token_indices = token_indices
        self.display = display
    
    def should_save_step(self, step: int) -> bool:
        """Check if this step should be saved."""
        return self.save and step in self.save_steps
    
    def get_save_path(self, step: int) -> Optional[str]:
        """Get the save path for a specific step."""
        if not self.save_dir:
            return None
        self.save_dir.mkdir(parents=True, exist_ok=True)
        return str(self.save_dir / f"step_{step:03d}.png")


def visualize_attention_store(
    prompt: str,
    attention_store: AttentionStore,
    tokenizer,
    attention_res: int = 16,
    token_indices: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    display: bool = False,
    orig_image=None,
) -> None:
    """
    Visualize cross-attention maps from an attention store.
    
    Args:
        prompt: The text prompt used
        attention_store: AttentionStore containing attention maps
        tokenizer: CLIP tokenizer
        attention_res: Resolution of attention maps (16, 32, 64, etc.)
        token_indices: Which token indices to visualize
        save_path: Path to save visualization to (if None, only displays)
        display: Whether to display visualization inline
        orig_image: Original image to overlay (optional)
    """
    visualize_cross_attention_maps(
        prompt=prompt,
        attention_store=attention_store,
        tokenizer=tokenizer,
        res=attention_res,
        from_where=["up", "down", "mid"],
        select=0,
        token_indices=token_indices or [],
        orig_image=orig_image,
        save_path=save_path,
        display_image=display,
    )
