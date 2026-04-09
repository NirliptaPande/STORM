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
    xs = torch.arange(w, device=attn.device, dtype=torch.float32)  # column indices
    ys = torch.arange(h, device=attn.device, dtype=torch.float32)  # row indices
    cx = (attn.sum(dim=0) * xs).sum()  # x centroid (column)
    cy = (attn.sum(dim=1) * ys).sum()  # y centroid (row)
    return cx, cy


def compute_gradient(attn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute spatial gradient of attention map via central differences.
    Edges handled with one-sided differences.
    attn: (16, 16)
    Returns grad_x, grad_y each (16, 16)
    """
    # grad_x: gradient along columns (x direction)
    grad_x = torch.zeros_like(attn)
    grad_x[:, 1:-1] = (attn[:, 2:] - attn[:, :-2]) / 2.0   # central diff
    grad_x[:, 0]    = attn[:, 1] - attn[:, 0]                # forward diff
    grad_x[:, -1]   = attn[:, -1] - attn[:, -2]              # backward diff

    # grad_y: gradient along rows (y direction)
    grad_y = torch.zeros_like(attn)
    grad_y[1:-1, :] = (attn[2:, :] - attn[:-2, :]) / 2.0
    grad_y[0, :]    = attn[1, :] - attn[0, :]
    grad_y[-1, :]   = attn[-1, :] - attn[-2, :]

    return grad_x, grad_y


def compute_divergence(vx: torch.Tensor, vy: torch.Tensor) -> torch.Tensor:
    """
    Compute divergence of vector field (vx, vy) via central differences.
    vx, vy: (16, 16)
    Returns div: (16, 16)
    """
    # d(vx)/dx
    dvx_dx = torch.zeros_like(vx)
    dvx_dx[:, 1:-1] = (vx[:, 2:] - vx[:, :-2]) / 2.0
    dvx_dx[:, 0]    = vx[:, 1] - vx[:, 0]
    dvx_dx[:, -1]   = vx[:, -1] - vx[:, -2]

    # d(vy)/dy
    dvy_dy = torch.zeros_like(vy)
    dvy_dy[1:-1, :] = (vy[2:, :] - vy[:-2, :]) / 2.0
    dvy_dy[0, :]    = vy[1, :] - vy[0, :]
    dvy_dy[-1, :]   = vy[-1, :] - vy[-2, :]

    return dvx_dx + dvy_dy


def build_neumann_laplacian(h: int, w: int, device: torch.device) -> torch.Tensor:
    """
    Build discrete Laplacian matrix with Neumann (zero-flux) boundary conditions.
    Size: (h*w, h*w)
    Neumann BC: attention mass stays on grid, no flux out of boundary.
    """
    n = h * w
    L = torch.zeros(n, n, device=device)

    for i in range(h):
        for j in range(w):
            idx = i * w + j
            # number of valid neighbors - this is what Neumann BC changes
            # vs Dirichlet: instead of fixing boundary values,
            # we just don't add flux terms for missing neighbors
            neighbors = 0

            if i > 0:      # up
                L[idx, (i-1)*w + j] = 1
                neighbors += 1
            if i < h-1:    # down
                L[idx, (i+1)*w + j] = 1
                neighbors += 1
            if j > 0:      # left
                L[idx, i*w + (j-1)] = 1
                neighbors += 1
            if j < w-1:    # right
                L[idx, i*w + (j+1)] = 1
                neighbors += 1

            L[idx, idx] = -neighbors  # negative sum of neighbors

    return L


def solve_poisson(b: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Solve Lf = b with Neumann BC.
    Neumann Laplacian is singular (null space = constant functions),
    so we fix one value to make the system uniquely solvable —
    specifically we pin f[0,0] = A_src[0,0] to anchor the solution.
    b: (16, 16) divergence field
    Returns f: (16, 16)
    """
    h, w = b.shape
    n = h * w

    L = build_neumann_laplacian(h, w, device)  # (256, 256)
    b_flat = b.flatten()                        # (256,)

    # Neumann Laplacian has a null space (adding a constant to f is also a solution)
    # Fix this by pinning one equation: f[0] = b[0]
    # i.e. replace first row of L with identity row
    L[0, :] = 0.0
    L[0, 0] = 1.0
    b_flat = b_flat.clone()
    b_flat[0] = 0.0  # anchor value — will renormalize anyway

    # Solve the linear system
    # lstsq handles any remaining near-singularity gracefully
    f_flat = torch.linalg.lstsq(L, b_flat.unsqueeze(1)).solution.squeeze(1)

    return f_flat.reshape(h, w)


def poisson_attention_push(
    A_src: torch.Tensor,
    A_ref: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """
    Replace STORM's OT-based attention repositioning with Poisson blending.
    
    Args:
        A_src:     (16, 16) source attention map (e.g. 'car'), normalized
        A_ref:     (16, 16) reference attention map (e.g. 'elephant'), normalized
        direction: (2,) fixed unit vector from prompt, e.g. tensor([-1., 0.]) for 'left'
    
    Returns:
        f: (16, 16) updated attention map, normalized
    """
    device = A_src.device

    # Step 1 — normalize inputs to be safe
    A_src = A_src / (A_src.sum() + 1e-8)
    A_ref = A_ref / (A_ref.sum() + 1e-8)

    # Step 2 — compute centroids and magnitude
    cx_src, cy_src = compute_centroid(A_src)
    cx_ref, cy_ref = compute_centroid(A_ref)
    magnitude = torch.sqrt((cx_ref - cx_src)**2 + (cy_ref - cy_src)**2)

    # Step 3 — compute gradient of source attention
    grad_x, grad_y = compute_gradient(A_src)  # each (16, 16)

    # Step 4 — build guidance field v
    # direction is (2,): direction[0] = dx, direction[1] = dy
    dx, dy = direction[0], direction[1]

    # push term: magnitude * d * A_src * (1 - A_ref)
    # this is the spatial part — high where source is, low where reference is
    push = magnitude * A_src * (1 - A_ref)  # (16, 16) scalar field

    # full vector field v
    vx = grad_x + dx * push  # (16, 16)
    vy = grad_y + dy * push  # (16, 16)

    # Step 5 — compute divergence of v (RHS of Poisson equation)
    b = compute_divergence(vx, vy)  # (16, 16)

    # Step 6 — solve Poisson equation: Lf = b
    f = solve_poisson(b, device)  # (16, 16)

    # Step 7 — post-process: clamp negatives, renormalize
    f = f.clamp(min=0)
    f = f / (f.sum() + 1e-8)

    return f


def get_direction_from_prompt(prompt: str) -> torch.Tensor:
    """
    Parse fixed direction vector from prompt.
    Returns unit vector (2,): [dx, dy]
    dx: positive = right, negative = left
    dy: positive = down, negative = up (image coordinates)
    """
    if "left" in prompt:
        return torch.tensor([-1., 0.])
    elif "right" in prompt:
        return torch.tensor([1., 0.])
    elif "above" in prompt or "top" in prompt:
        return torch.tensor([0., -1.])
    elif "below" in prompt:
        return torch.tensor([0., 1.])
    else:
        return torch.tensor([0., 0.])  # no spatial relationship