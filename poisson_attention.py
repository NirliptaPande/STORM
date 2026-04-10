import torch
import torch.nn.functional as F
from typing import Tuple


def compute_centroid(attn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute centroid of a normalized 16x16 attention map.
    Returns (cx, cy) in grid coordinates [0, 15].
    attn: (16, 16), assumed to sum to 1
    """
    h, w = attn.shape
    xs = torch.arange(w, device=attn.device, dtype=torch.float32)
    ys = torch.arange(h, device=attn.device, dtype=torch.float32)
    cx = (attn.sum(dim=0) * xs).sum()
    cy = (attn.sum(dim=1) * ys).sum()
    return cx, cy


def compute_storm_cost_matrix(
    A_src: torch.Tensor,
    A_ref: torch.Tensor,
    direction: str,
    w: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute STORM's ST Cost matrix on the 16x16 grid.
    Directly mirrors _compute_cost_function in pipeline.py.

    C_ij = A_ref_ij * delta_ij(desired, restricted)

    where delta encodes:
    - low cost on the desired side of the reference centroid
    - high cost on the restricted side

    Args:
        A_src:     (16, 16) source attention, normalized
        A_ref:     (16, 16) reference attention, normalized
        direction: one of 'left', 'right', 'above', 'below'
        w:         progressive weight (STORM's omega), controls asymmetry strength
        eps:       numerical stability

    Returns:
        C: (16, 16) scalar cost field
    """
    device = A_src.device
    h, w_grid = A_src.shape  # 16, 16

    cx_ref, cy_ref = compute_centroid(A_ref)

    # coordinate grids
    xs = torch.arange(w_grid, device=device, dtype=torch.float32).unsqueeze(0)  # (1, 16)
    ys = torch.arange(h,      device=device, dtype=torch.float32).unsqueeze(1)  # (16, 1)

    # directional delta values — distances from each point to reference centroid
    # delta_des: positive when on the desired side → low cost
    # delta_res: positive when on the restricted side → high cost
    if direction == 'left':
        delta_des = cx_ref - xs   # positive when left of ref centroid ✓
        delta_res = xs - cx_ref   # positive when right of ref centroid ✗
        delta_des = delta_des.expand(h, -1)
        delta_res = delta_res.expand(h, -1)
    elif direction == 'right':
        delta_des = xs - cx_ref
        delta_res = cx_ref - xs
        delta_des = delta_des.expand(h, -1)
        delta_res = delta_res.expand(h, -1)
    elif direction == 'above':
        delta_des = cy_ref - ys   # positive when above ref centroid ✓
        delta_res = ys - cy_ref
        delta_des = delta_des.expand(-1, w_grid)
        delta_res = delta_res.expand(-1, w_grid)
    elif direction == 'below':
        delta_des = ys - cy_ref
        delta_res = cy_ref - ys
        delta_des = delta_des.expand(-1, w_grid)
        delta_res = delta_res.expand(-1, w_grid)
    else:
        # no spatial relationship — uniform cost, just prevent overlap
        return A_ref

    # indicator: only apply when positive (on that side of centroid)
    ind_des = (delta_des > 0).float()
    ind_res = (delta_res > 0).float()

    # STORM's delta function (Eq. 2 in paper):
    # 1 / (w * (delta_des + eps)) when on desired side → low cost
    # w * (delta_res + eps)        when on restricted side → high cost
    delta = (
        ind_des / (w * (delta_des + eps))
        + ind_res * w * (delta_res + eps)
    )

    # ST Cost: weight by reference attention (non-overlap)
    # C_ij = A_ref_ij * delta_ij  (Eq. 3 in paper)
    C = A_ref * delta  # (16, 16)

    return C


def build_neumann_laplacian(h: int, w: int, device: torch.device) -> torch.Tensor:
    """
    Build discrete Laplacian with Neumann (zero-flux) boundary conditions.
    Size: (h*w, h*w)
    """
    n = h * w
    L = torch.zeros(n, n, device=device)

    for i in range(h):
        for j in range(w):
            idx = i * w + j
            neighbors = 0
            if i > 0:
                L[idx, (i-1)*w + j] = 1
                neighbors += 1
            if i < h-1:
                L[idx, (i+1)*w + j] = 1
                neighbors += 1
            if j > 0:
                L[idx, i*w + (j-1)] = 1
                neighbors += 1
            if j < w-1:
                L[idx, i*w + (j+1)] = 1
                neighbors += 1
            L[idx, idx] = -neighbors

    return L


def solve_poisson(C: torch.Tensor) -> torch.Tensor:
    """
    Solve the Poisson equation: ∇²f = C
    with Neumann boundary conditions.

    C is STORM's cost matrix — used directly as the RHS.
    High cost regions in C repel attention mass,
    low cost regions attract it, exactly as in STORM's cost matrix.

    Args:
        C: (16, 16) STORM cost matrix

    Returns:
        f: (16, 16) target attention map
    """
    device = C.device
    h, w = C.shape  # 16, 16

    L = build_neumann_laplacian(h, w, device)  # (256, 256)
    b = C.flatten()                             # (256,)

    # Pin one value to resolve null space (Neumann BC → L is singular)
    L[0, :] = 0.0
    L[0, 0] = 1.0
    b = b.clone()
    b[0] = 0.0

    f = torch.linalg.lstsq(L, b.unsqueeze(1)).solution.squeeze(1)
    return f.reshape(h, w)


def poisson_attention_push(
    A_src: torch.Tensor,
    A_ref: torch.Tensor,
    direction: str,
    w: float = 1.0,
) -> torch.Tensor:
    """
    Replace STORM's OT with a single Poisson solve.

    Uses STORM's cost matrix directly as the RHS of ∇²f = C,
    producing f in one shot — no iterative Sinkhorn needed.

    Args:
        A_src:     (16, 16) source attention (e.g. 'cake'), normalized
        A_ref:     (16, 16) reference attention (e.g. 'suitcase'), normalized
        direction: 'left', 'right', 'above', 'below'
        w:         STORM's progressive weight omega, passed in from timestep

    Returns:
        f: (16, 16) target attention map, normalized
    """
    # normalize
    A_src = A_src / (A_src.sum() + 1e-8)
    A_ref = A_ref / (A_ref.sum() + 1e-8)

    # Step 1 — STORM cost matrix as RHS
    C = compute_storm_cost_matrix(A_src, A_ref, direction, w=w)  # (16, 16)

    # Step 2 — single Poisson solve: ∇²f = C
    f = solve_poisson(C)  # (16, 16)

    # Step 3 — post-process
    f = f.clamp(min=0)
    f = f / (f.sum() + 1e-8)

    return f


def get_direction_from_prompt(prompt: str) -> str:
    """Parse spatial direction from prompt string."""
    if "left"  in prompt: return "left"
    if "right" in prompt: return "right"
    if "above" in prompt or "top" in prompt: return "above"
    if "below" in prompt: return "below"
    return "none"