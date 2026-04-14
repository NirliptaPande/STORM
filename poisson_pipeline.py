import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from packaging import version
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import deprecate, is_accelerate_available, logging, replace_example_docstring
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

from utils.gaussian_smoothing import GaussianSmoothing
from utils.ptp_utils import AttentionStore, aggregate_attention
from utils.vis_utils import visualize_cross_attention_maps

logger = logging.get_logger(__name__)


class StormPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion,
    with Poisson-equation-based attention guidance replacing STORM's Sinkhorn OT.
    """

    # ------------------------------------------------------------------
    # Utility helpers (unchanged from STORM)
    # ------------------------------------------------------------------

    def _compute_centroid_2d(self, attn_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = torch.arange(attn_map.shape[0], device=attn_map.device).float()
        x_centroid = (attn_map.sum(dim=0) * indices).sum() / (attn_map.sum() + 1e-8)
        y_centroid = (attn_map.sum(dim=1) * indices).sum() / (attn_map.sum() + 1e-8)
        return x_centroid, y_centroid

    def _exp_cost_weight(self, t_values: Union[int, float, torch.Tensor],
                         k: float = 0.4, W_max: int = 100) -> torch.Tensor:
        if not isinstance(t_values, torch.Tensor):
            t_values = torch.tensor(t_values, device=self.device)
        return 1 + (W_max - 1) * (1 - torch.exp(-k * t_values))

    def _normalize_attention_map(self, attn_map: torch.Tensor) -> torch.Tensor:
        return attn_map / (attn_map.sum() + 1e-8)

    def _apply_smoothing(self, attn_map: torch.Tensor,
                         kernel_size: int, sigma: float) -> torch.Tensor:
        smoothing = GaussianSmoothing(
            channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
        ).to(attn_map.device)
        padded = F.pad(attn_map.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
        return smoothing(padded).squeeze(0).squeeze(0)

    def _get_attention_maps_for_tokens(
        self,
        attention_for_text: torch.Tensor,
        indices: List[Optional[int]]
    ) -> Optional[torch.Tensor]:
        valid = [i for i in indices if i is not None]
        if not valid:
            return None
        sel = attention_for_text[:, :, valid]
        return sel.mean(dim=-1) if len(valid) > 1 else sel.squeeze(-1)

    # ------------------------------------------------------------------
    # Cost field  (replaces _compute_cost_function, stays (H,W))
    # ------------------------------------------------------------------

    def _compute_cost_field(self, attn: torch.Tensor, direction: str,
                            w: float = 100,
                            use_distance: bool = False) -> torch.Tensor:
        """
        Returns a (H, W) cost field. High = bad location, Low = good location.
        `attn` is the *other* object's attention map (cross-referenced, same as STORM).
        Never expands to (N, N).
        """
        H, W = attn.shape
        cx, cy = self._compute_centroid_2d(attn)

        x = torch.arange(W, device=attn.device).float().unsqueeze(0)  # (1, W)
        y = torch.arange(H, device=attn.device).float().unsqueeze(1)  # (H, 1)

        if direction == 'left':
            cost_factor = torch.where(
                x < cx,
                1 / (w * (cx - x + 1e-8)),
                w * (x - cx + 1e-8))
        elif direction == 'right':
            cost_factor = torch.where(
                x > cx,
                1 / (w * (x - cx + 1e-8)),
                w * (cx - x + 1e-8))
        elif direction == 'above':
            cost_factor = torch.where(
                y < cy,
                1 / (w * (cy - y + 1e-8)),
                w * (y - cy + 1e-8)
            ).transpose(0, 1)
        elif direction == 'below':
            cost_factor = torch.where(
                y > cy,
                1 / (w * (y - cy + 1e-8)),
                w * (cy - y + 1e-8)
            ).transpose(0, 1)
        else:
            cost_factor = torch.ones_like(attn)

        cost_field = attn * cost_factor   # (H, W)

        if use_distance:
            dist = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            dist_norm = dist / (dist.mean() + 1e-8)
            cost_field = cost_field * dist_norm

        return cost_field

    # ------------------------------------------------------------------
    # Adjective cost field  (replaces _compute_cost_function_adj)
    # ------------------------------------------------------------------

    def _compute_cost_field_adj(self, sub_attn_flat: torch.Tensor,
                                sub_attn_adj_flat: torch.Tensor,
                                alpha: float = 0.5,
                                beta: float = 0.5) -> torch.Tensor:
        """Returns a (N,) cost field for adjective guidance."""
        N = sub_attn_flat.size(0)
        device = sub_attn_flat.device
        pos = torch.arange(N, device=device).float()
        centroid_adj = (pos * sub_attn_adj_flat).sum() / (sub_attn_adj_flat.sum() + 1e-8)
        pos_cost = (pos - centroid_adj).abs()
        val_cost = (sub_attn_flat - sub_attn_adj_flat).abs()
        cost_field = alpha * pos_cost + beta * val_cost
        mx = cost_field.max()
        return cost_field / mx if mx > 1e-6 else torch.zeros_like(cost_field)

    # ------------------------------------------------------------------
    # Laplacian with Neumann boundary conditions
    # ------------------------------------------------------------------

    @staticmethod
    def _build_laplacian_neumann(H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        (N, N) graph Laplacian for H×W grid, Neumann (zero normal-derivative) BC.
        Interior: degree 4, edge: degree 3, corner: degree 2.
        """
        N = H * W
        L = torch.zeros(N, N, device=device)

        def idx(r, c): return r * W + c

        for r in range(H):
            for c in range(W):
                i = idx(r, c)
                nbrs = []
                if r > 0:     nbrs.append(idx(r - 1, c))
                if r < H - 1: nbrs.append(idx(r + 1, c))
                if c > 0:     nbrs.append(idx(r, c - 1))
                if c < W - 1: nbrs.append(idx(r, c + 1))
                L[i, i] = -len(nbrs)
                for j in nbrs:
                    L[i, j] = 1.0
        return L
    
    # ------------------------------------------------------------------
    # Gaussian target  (used by loss option 8.2)
    # ------------------------------------------------------------------   
    @staticmethod
    def _make_gaussian_target(H: int, W: int, cx: float, cy: float,
                               sigma: float, device: torch.device) -> torch.Tensor:
        """
        Returns a (H, W) normalised Gaussian centred at (cx, cy).
        cx, cy are in pixel coordinates (can be fractional).
        """
        xs = torch.arange(W, device=device).float()
        ys = torch.arange(H, device=device).float()
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        g = torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        return g / (g.sum() + 1e-8)
    # ------------------------------------------------------------------
    # Poisson RHS
    # ------------------------------------------------------------------

    @staticmethod
    def _cost_field_to_rhs(cost_field: torch.Tensor) -> torch.Tensor:
        """
        f = -(cost_field - mean(cost_field))
        Mean-centering → sum(f)=0 (Neumann compatibility condition).
        Negation → high cost = sink in a', low cost = source.
        """
        f = cost_field.flatten()
        return -(f - f.mean())

    # ------------------------------------------------------------------
    # Poisson solve
    # ------------------------------------------------------------------

    def _solve_poisson(self, rhs: torch.Tensor, mass: torch.Tensor,
                       H: int, W: int) -> torch.Tensor:
        """
        Solve  [L ; 1^T] a' = [rhs ; mass]  via least squares.
        Clamp to ≥0 and renormalise to restore mass.
        """
        N = H * W
        device = rhs.device
        with torch.no_grad():
            L    = self._build_laplacian_neumann(H, W, device)
            ones = torch.ones(1, N, device=device)
            A    = torch.cat([L, ones], dim=0)                  # (N+1, N)
            b    = torch.cat([rhs, mass.unsqueeze(0)], dim=0)   # (N+1,)

            a_prime = torch.linalg.lstsq(A, b).solution         # (N,)

            a_prime = a_prime.clamp(min=0)
            s = a_prime.sum()
            if s > 1e-8:
                a_prime = a_prime * (mass / s)
        return a_prime.reshape(H, W)

    # ------------------------------------------------------------------
    # Main loss  (replaces _storm_loss)
    # ------------------------------------------------------------------

    def _poisson_loss(self,
                      time_step: int,
                      attention_maps: torch.Tensor,
                      indices_to_alter_total: List[List[Optional[int]]],
                      smooth_attentions: bool = False,
                      sigma: float = 0.5,
                      kernel_size: int = 3,
                      normalize_eot: bool = False,
                      use_distance: bool = False,
                      centroid: Optional[List[int]] = None
                      ) -> Tuple[torch.Tensor, List[float]]:

        device = attention_maps.device

        # 1. Prepare attention maps
        prompt_str = self.prompt[0] if isinstance(self.prompt, list) else self.prompt
        last_idx   = len(self.tokenizer(prompt_str)['input_ids']) - 1 if normalize_eot else -1
        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text = F.softmax(attention_for_text * 100, dim=-1)

        shifted = [[idx - 1 if idx is not None else None for idx in sl]
                   for sl in indices_to_alter_total]
        subject_indices, adjective_indices = shifted

        # 2. Parse spatial condition
        is_spatial      = any(w in prompt_str for w in ["top", "above", "below"])
        is_horizontal   = any(w in prompt_str for w in ["left", "right"])

        sub_idx, obj_idx = 0, 1

        # 3. Extract attention maps
        sub_attn = self._get_attention_maps_for_tokens(
            attention_for_text, [subject_indices[sub_idx]])
        obj_attn = self._get_attention_maps_for_tokens(
            attention_for_text, [subject_indices[obj_idx]])

        if sub_attn is None or obj_attn is None:
            logger.warning("Sub/obj attention maps are None. Returning zero loss.")
            return torch.tensor(0.0, device=device), [0.0, 0.0]

        sub_attn_adj = self._get_attention_maps_for_tokens(
            attention_for_text, [adjective_indices[sub_idx]])
        obj_attn_adj = self._get_attention_maps_for_tokens(
            attention_for_text, [adjective_indices[obj_idx]])

        # 4. Smooth
        if smooth_attentions:
            sub_attn = self._apply_smoothing(sub_attn, kernel_size, sigma)
            obj_attn = self._apply_smoothing(obj_attn, kernel_size, sigma)

        # 5. Reshape to 2-D, normalise
        res    = attention_maps.shape[1]
        sub_2d = self._normalize_attention_map(sub_attn).reshape(res, res)
        obj_2d = self._normalize_attention_map(obj_attn).reshape(res, res)
        H, W   = sub_2d.shape

        # Coordinate (same as STORM, used for threshold gating)
        cx_sub, cy_sub = self._compute_centroid_2d(sub_2d)
        cx_obj, cy_obj = self._compute_centroid_2d(obj_2d)
        if is_horizontal:
            coordinate = [(cx_obj - cx_sub).clamp(min=0).item() / 10,
                          (cy_obj - cy_sub).clamp(min=0).item() / 10]
        else:
            coordinate = [(cx_sub - cx_obj).clamp(min=0).item() / 10,
                          (cy_sub - cy_obj).clamp(min=0).item() / 10]

        # 6. Time-varying weight
        w = self._exp_cost_weight(time_step)

        # 7. Cost fields (cross-referenced, same logic as STORM)
        if is_spatial:
            cf_sub = self._compute_cost_field(obj_2d, 'above', w=w, use_distance=use_distance)
            cf_obj = self._compute_cost_field(sub_2d, 'below', w=w, use_distance=use_distance)
        elif is_horizontal:
            cf_sub = self._compute_cost_field(obj_2d, 'left',  w=w, use_distance=use_distance)
            cf_obj = self._compute_cost_field(sub_2d, 'right', w=w, use_distance=use_distance)
        else:
            cf_sub = self._compute_cost_field(obj_2d, 'obj', w=w, use_distance=use_distance)
            cf_obj = self._compute_cost_field(sub_2d, 'obj', w=w, use_distance=use_distance)

        # There are a bunch of options for the loss here — we found direct weighted loss to work best, but the Poisson-solve-based spatial loss is also interesting and worth exploring further in future work.
        
        # # 8.1 Poisson solve → target attention maps
        target_sub = self._solve_poisson(
            self._cost_field_to_rhs(cf_sub), sub_2d.sum(), H, W)
        target_obj = self._solve_poisson(
            self._cost_field_to_rhs(cf_obj), obj_2d.sum(), H, W)

        # # 8.2 Gaussian target
        # target_sub = self._make_gaussian_target(H, W, cx=(0 + cx_obj.item()) / 2, cy=H*0.5, sigma=3.0, device=device)
        # target_obj = self._make_gaussian_target(H, W, cx=(W + cx_sub.item()) / 2, cy=H*0.5, sigma=3.0, device=device)
        # loss_sub = F.kl_div((sub_2d + 1e-8).log(), target_sub + 1e-8, reduction='sum')
        # loss_obj = F.kl_div((obj_2d + 1e-8).log(), target_obj + 1e-8, reduction='sum')
         

        # # 9.1 Spatial MSE loss for 8.1 Poisson solution
        loss_sub = F.mse_loss(sub_2d, target_sub) * 100
        loss_obj = F.mse_loss(obj_2d, target_obj) * 100
        
        # # 9.2 Direct L2 loss to a cost-weighted target
        # loss_sub = F.kl_div((sub_2d + 1e-8).log(), target_sub + 1e-8, reduction='sum')
        # loss_obj = F.kl_div((obj_2d + 1e-8).log(), target_obj + 1e-8, reduction='sum')
        
        # # 9.3 Direct weighted loss — penalize attention in high-cost regions
        # loss_sub = (sub_2d * cf_sub).sum()
        # loss_obj = (obj_2d * cf_obj).sum()
        
        # # 9.4 Centroid baseline, no weights
        # margin = 2.0
        # loss_sub = F.relu(cx_sub - cx_obj + margin)
        # loss_obj = F.relu(cx_sub - cx_obj + margin)
            # print(f"sub_2d sum: {sub_2d.sum():.4f}, target_sub sum: {target_sub.sum():.4f}, "
        #     f"target_sub max: {target_sub.max():.4f}, target_sub min: {target_sub.min():.4f}")
        # print(f"obj_2d sum: {obj_2d.sum():.4f}, target_obj sum: {target_obj.sum():.4f}, "
        #     f"target_obj max: {target_obj.max():.4f}, target_obj min: {target_obj.min():.4f}")   
             
        # 10. Adjective loss
        loss_adj  = torch.tensor(0.0, device=device)
        adj_count = 0
        if sub_attn_adj is not None:
            sf  = sub_2d.flatten()
            saf = self._normalize_attention_map(sub_attn_adj).flatten()
            cf  = self._compute_cost_field_adj(sf, saf)
            loss_adj  += (cf * (sf - saf).pow(2)).mean()
            adj_count += 1
        if obj_attn_adj is not None:
            of  = obj_2d.flatten()
            oaf = self._normalize_attention_map(obj_attn_adj).flatten()
            cf  = self._compute_cost_field_adj(of, oaf)
            loss_adj  += (cf * (of - oaf).pow(2)).mean()
            adj_count += 1
        if adj_count > 1:
            loss_adj = loss_adj / adj_count

        loss = loss_sub + loss_obj + loss_adj
        return loss, coordinate

    # ------------------------------------------------------------------
    # Public wrapper  (identical signature to STORM's _compute_loss_from_ot)
    # ------------------------------------------------------------------

    def _compute_loss_from_ot(self,
                              time_step: int,
                              attention_store: AttentionStore,
                              indices_to_alter_total: List[int],
                              attention_res: int = 16,
                              smooth_attentions: bool = False,
                              sigma: float = 0.5,
                              kernel_size: int = 3,
                              normalize_eot: bool = False,
                              use_distance: bool = False,
                              centroid: List[int] = None
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)

        loss, coordinate = self._poisson_loss(
            time_step=time_step,
            attention_maps=attention_maps,
            indices_to_alter_total=indices_to_alter_total,
            normalize_eot=normalize_eot,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            use_distance=use_distance,
            centroid=centroid)

        return loss, coordinate

    # ------------------------------------------------------------------
    # Latent update  (unchanged from STORM)
    # ------------------------------------------------------------------

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor,
                       step_size: float, max_norm: float = 1.0) -> torch.Tensor:
        grad = torch.autograd.grad(loss.requires_grad_(True), [latents],
                                   retain_graph=True)[0]
        g_norm = torch.norm(grad)
        if g_norm > max_norm:
            grad = grad * (max_norm / g_norm)
        return latents - step_size * grad

    # ------------------------------------------------------------------
    # Iterative refinement  (unchanged from STORM)
    # ------------------------------------------------------------------

    def _perform_iterative_refinement_step_spatial(self,
                                                   time_step: int,
                                                   latents: torch.Tensor,
                                                   indices_to_alter_total: List[int],
                                                   loss: torch.Tensor,
                                                   threshold: float,
                                                   text_embeddings: torch.Tensor,
                                                   text_input,
                                                   attention_store: AttentionStore,
                                                   step_size: float,
                                                   t: int,
                                                   attention_res: int = 16,
                                                   smooth_attentions: bool = True,
                                                   sigma: float = 0.5,
                                                   kernel_size: int = 3,
                                                   max_refinement_steps: int = 30,
                                                   normalize_eot: bool = False,
                                                   use_distance: bool = False,
                                                   centroid: List[int] = None
                                                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        iteration   = 0
        target_loss = max(0, 1.0 - threshold)

        while loss > target_loss:
            iteration += 1
            latents = latents.clone().detach().requires_grad_(True)
            noise_pred_text = self.unet(
                latents, t,
                encoder_hidden_states=text_embeddings[1].unsqueeze(0)
            ).sample
            self.unet.zero_grad()

            loss, _ = self._compute_loss_from_ot(
                time_step=time_step,
                attention_store=attention_store,
                indices_to_alter_total=indices_to_alter_total,
                smooth_attentions=smooth_attentions,
                sigma=sigma,
                kernel_size=kernel_size,
                attention_res=attention_res,
                normalize_eot=False,
                use_distance=use_distance,
                centroid=centroid)

            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)

            with torch.no_grad():
                noise_pred_uncond = self.unet(
                    latents, t,
                    encoder_hidden_states=text_embeddings[0].unsqueeze(0)
                ).sample
                noise_pred_text = self.unet(
                    latents, t,
                    encoder_hidden_states=text_embeddings[1].unsqueeze(0)
                ).sample

            latents = latents.clone().detach().requires_grad_(True)
            noise_pred_text = self.unet(
                latents, t,
                encoder_hidden_states=text_embeddings[1].unsqueeze(0)
            ).sample

            if iteration >= max_refinement_steps:
                print(f'\t Exceeded max refinement steps ({max_refinement_steps})!')
                break

        self.unet.zero_grad()
        return loss, latents

    # ------------------------------------------------------------------
    # __call__  (denoising loop — faithful to STORM's original)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self,
                 prompt: Union[str, List[str]],
                 attention_store: AttentionStore,
                 indices_to_alter: List[int],
                 attention_res: int = 16,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 num_inference_steps: int = 50,
                 guidance_scale: float = 7.5,
                 negative_prompt: Optional[Union[str, List[str]]] = None,
                 num_images_per_prompt: Optional[int] = 1,
                 eta: float = 0.0,
                 generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                 latents: Optional[torch.FloatTensor] = None,
                 prompt_embeds: Optional[torch.FloatTensor] = None,
                 negative_prompt_embeds: Optional[torch.FloatTensor] = None,
                 output_type: Optional[str] = "pil",
                 return_dict: bool = True,
                 callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                 callback_steps: int = 1,
                 cross_attention_kwargs: Optional[Dict[str, Any]] = None,
                 max_iter_to_alter: Optional[int] = 25,
                 run_standard_sd: bool = False,
                 thresholds: Optional[dict] = None,
                 scale_factor: int = 20,
                 scale_range: Tuple[float, float] = (1., 0.5),
                 smooth_attentions: bool = True,
                 sigma: float = 0.5,
                 kernel_size: int = 3,
                 sd_2_1: bool = False,
                 use_distance: bool = False,
                 attn_snapshot_steps: Optional[List[int]] = None,
                 attn_snapshot_dir: Optional[str] = None,
                 attn_snapshot_token_indices: Optional[List[int]] = None,
                 display_attention_maps: bool = False,
                 ):
        if thresholds is None:
            thresholds = {0: 0.9, 5: 0.95, 10: 0.99, 15: 0.995, 20: 0.999}

        # Standard SD setup
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width  = width  or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(prompt, height, width, callback_steps,
                          negative_prompt, prompt_embeds, negative_prompt_embeds)

        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_cfg = guidance_scale > 1.0

        prompt_embeds = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_cfg,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds)
        text_inputs = None

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents, height, width,
            prompt_embeds.dtype, device, generator, latents)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        scale_range = np.linspace(scale_range[0], scale_range[1],
                                  len(self.scheduler.timesteps))

        if max_iter_to_alter is None:
            max_iter_to_alter = len(self.scheduler.timesteps) + 1

        # Use prompt_str for the spatial condition check (handles list prompts)
        prompt_str = prompt[0] if isinstance(prompt, list) else prompt
        spatial_condition = 0 if ("left" in prompt_str or "right" in prompt_str) else 1
        snapshot_steps = set(attn_snapshot_steps or [])
        saved_snapshot_steps = set()
        default_token_indices = []
        for group in indices_to_alter:
            if isinstance(group, (list, tuple, set)):
                default_token_indices.extend([idx for idx in group if idx is not None])
            elif group is not None:
                default_token_indices.append(group)
        token_indices_for_viz = attn_snapshot_token_indices or default_token_indices

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                with torch.enable_grad():
                    latents = latents.clone().detach().requires_grad_(True)

                    noise_pred_text = self.unet(
                        latents, t,
                        encoder_hidden_states=prompt_embeds[1].unsqueeze(0),
                        cross_attention_kwargs=cross_attention_kwargs
                    ).sample
                    self.unet.zero_grad()

                    loss, coordinate = self._compute_loss_from_ot(
                        time_step=i,
                        attention_store=attention_store,
                        indices_to_alter_total=indices_to_alter,
                        attention_res=attention_res,
                        smooth_attentions=smooth_attentions,
                        sigma=sigma,
                        kernel_size=kernel_size,
                        normalize_eot=sd_2_1,
                        use_distance=use_distance)

                    if i in snapshot_steps and i not in saved_snapshot_steps:
                        save_path = None
                        if attn_snapshot_dir:
                            save_path = str(Path(attn_snapshot_dir) / f"step_{i:03d}.png")
                        visualize_cross_attention_maps(
                            prompt=prompt_str,
                            attention_store=attention_store,
                            tokenizer=self.tokenizer,
                            res=attention_res,
                            from_where=["up", "down", "mid"],
                            select=0,
                            token_indices=token_indices_for_viz,
                            orig_image=None,
                            save_path=save_path,
                            display_image=display_attention_maps,
                        )
                        saved_snapshot_steps.add(i)

                    if not run_standard_sd and len(indices_to_alter[0]) == 2:
                        if (i in thresholds.keys()
                                and loss > 1. - thresholds[i]
                                and 0 < i < 25
                                and coordinate[spatial_condition] > 0):
                            del noise_pred_text
                            torch.cuda.empty_cache()

                            loss, latents = self._perform_iterative_refinement_step_spatial(
                                time_step=i,
                                latents=latents,
                                indices_to_alter_total=indices_to_alter,
                                loss=loss,
                                threshold=thresholds[i],
                                text_embeddings=prompt_embeds,
                                text_input=text_inputs,
                                attention_store=attention_store,
                                step_size=scale_factor * np.sqrt(scale_range[i]),
                                t=t,
                                attention_res=attention_res,
                                smooth_attentions=smooth_attentions,
                                sigma=sigma,
                                kernel_size=kernel_size,
                                normalize_eot=sd_2_1,
                                use_distance=use_distance)

                        if 0 < i < 25:
                            torch.cuda.empty_cache()

                            loss, _ = self._compute_loss_from_ot(
                                time_step=i,
                                attention_store=attention_store,
                                indices_to_alter_total=indices_to_alter,
                                smooth_attentions=smooth_attentions,
                                sigma=sigma,
                                kernel_size=kernel_size,
                                attention_res=attention_res,
                                normalize_eot=sd_2_1,
                                use_distance=use_distance)

                            if loss != 0:
                                latents = self._update_latent(
                                    latents=latents, loss=loss,
                                    step_size=scale_factor * np.sqrt(scale_range[i]))

                            print(f'Iteration {i} | Loss: {loss:0.4f}')

                # Scheduler step (outside enable_grad, same as STORM)
                torch.cuda.empty_cache()
                latent_model_input = (torch.cat([latents] * 2) if do_cfg else latents)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs
                ).sample

                if do_cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = (noise_pred_uncond
                                  + guidance_scale * (noise_pred_text - noise_pred_uncond))

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps
                        and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Post-processing
        image = self.decode_latents(latents)
        image, has_nsfw = self.run_safety_checker(image, device, prompt_embeds.dtype)
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        if not return_dict:
            return (image, has_nsfw)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw)