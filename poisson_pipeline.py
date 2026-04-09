from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from pipeline import StormPipeline
from poisson_attention import poisson_attention_push, get_direction_from_prompt
from utils.ptp_utils import AttentionStore, aggregate_attention


class PoissonPipeline(StormPipeline):
    """
    Replaces STORM's OT-based attention repositioning with Poisson blending.
    Inherits everything from StormPipeline — only the loss computation changes.
    """

    def _compute_loss_from_ot(
        self,
        time_step: int,
        attention_store: AttentionStore,
        indices_to_alter_total: List[int],
        attention_res: int = 16,
        smooth_attentions: bool = False,
        sigma: float = 0.5,
        kernel_size: int = 3,
        normalize_eot: bool = False,
        centroid: List[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Override _compute_loss_from_ot to use Poisson loss instead of OT loss.
        Interface is identical — __call__ and iterative refinement call this
        without knowing anything changed.
        """
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0,
        )

        losses, coordinate = self._poisson_loss(
            time_step=time_step,
            attention_maps=attention_maps,
            indices_to_alter_total=indices_to_alter_total,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot,
        )

        return losses, coordinate

    def _poisson_loss(
        self,
        time_step: int,
        attention_maps: torch.Tensor,
        indices_to_alter_total: List[List[Optional[int]]],
        smooth_attentions: bool = False,
        sigma: float = 0.5,
        kernel_size: int = 3,
        normalize_eot: bool = False,
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Compute loss using Poisson blending instead of OT.
        """
        device = attention_maps.device

        # ── 1. Prepare attention maps ─────────────────────────────────────────
        prompt_str = self.prompt[0] if isinstance(self.prompt, list) else self.prompt
        last_idx = (
            len(self.tokenizer(prompt_str)["input_ids"]) - 1
            if normalize_eot
            else -1
        )

        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text = F.softmax(attention_for_text * 100, dim=-1)

        # Shift indices — first token removed
        subject_indices, adjective_indices = [
            [idx - 1 if idx is not None else None for idx in sublist]
            for sublist in indices_to_alter_total
        ]

        # ── 2. Extract source and reference attention maps ────────────────────
        A_src = attention_for_text[:, :, subject_indices[0]].reshape(16, 16)
        A_ref = attention_for_text[:, :, subject_indices[1]].reshape(16, 16)

        A_src = A_src / (A_src.sum() + 1e-8)
        A_ref = A_ref / (A_ref.sum() + 1e-8)

        # ── 3. Optional smoothing ─────────────────────────────────────────────
        if smooth_attentions:
            A_src = self._apply_smoothing(A_src, kernel_size, sigma)
            A_ref = self._apply_smoothing(A_ref, kernel_size, sigma)

        # ── 4. Get fixed direction from prompt ────────────────────────────────
        direction = get_direction_from_prompt(prompt_str).to(device)

        if direction.sum() == 0:
            return torch.tensor(0.0, device=device), [0.0, 0.0]

        # ── 5. Poisson solve — get target attention map ───────────────────────
        f = poisson_attention_push(A_src, A_ref, direction)  # (16, 16)

        # ── 6. Loss: MSE between current attention and Poisson target ─────────
        loss = F.mse_loss(A_src, f)

        # ── 7. Coordinate for threshold check in __call__ ─────────────────────
        xs = torch.arange(16, device=device, dtype=torch.float32)
        cx_src = (A_src.sum(dim=0) * xs).sum()
        cx_ref = (A_ref.sum(dim=0) * xs).sum()
        coordinate = [(cx_ref - cx_src).abs().item(), 0.0]

        return loss, coordinate