"""Exact full-grid DTW-PA-JPE on the input device (no path normalization)."""

import torch


@torch.no_grad()
def pairwise_pa_jpe(x, y, wanted=None, pair_chunk_size=4096):
    """Match the legacy l2_dist_align(..., align_idx=None) cost.

    Align using ALL joints, then select ``wanted`` for scoring. The similarity
    transform includes scale and reflection correction, as in rigid_align.
    Chunk frame pairs to bound the temporary joint/SVD tensors.
    """
    if (
        x.ndim != 3 or y.ndim != 3 or x.shape[1:] != y.shape[1:]
        or x.shape[-1] != 3 or min(len(x), len(y), x.shape[1]) == 0
        or x.device != y.device
    ):
        raise ValueError("Expected nonempty [T, J, 3] inputs on the same device.")
    if pair_chunk_size < 1:
        raise ValueError("pair_chunk_size must be positive.")

    with torch.autocast(device_type=x.device.type, enabled=False):
        x, y = x.float(), y.float()
        if not (torch.isfinite(x).all() & torch.isfinite(y).all()):
            raise ValueError("DTW inputs must be finite.")
        xc = x - x.mean(dim=1, keepdim=True)
        yc = y - y.mean(dim=1, keepdim=True)
        variance = xc.square().mean(dim=1).sum(dim=-1)
        if (variance <= 0).any():
            raise ValueError("PA alignment is undefined for zero-variance poses.")

        rows, cols, joints = len(x), len(y), x.shape[1]
        costs = x.new_empty(rows * cols)
        for start in range(0, rows * cols, pair_chunk_size):
            stop = min(start + pair_chunk_size, rows * cols)
            ids = torch.arange(start, stop, device=x.device)
            ix, iy = ids // cols, ids % cols
            a, b = xc[ix], yc[iy]
            covariance = a.transpose(1, 2).bmm(b) / joints
            u, singular, vh = torch.linalg.svd(covariance, full_matrices=False)
            signs = torch.ones_like(singular)
            signs[:, -1] = torch.where(
                torch.linalg.det(u.bmm(vh)) < 0, -1.0, 1.0
            )
            rotation = (u * signs[:, None, :]).bmm(vh)
            scale = (singular * signs).sum(dim=-1) / variance[ix]
            # Centered coordinates cancel the translation in the residual.
            residual = scale[:, None, None] * a.bmm(rotation) - b
            if wanted is not None:
                residual = residual[:, wanted]
            costs[start:stop] = residual.norm(dim=-1).mean(dim=-1)
        return costs.reshape(rows, cols)


@torch.no_grad()
def dtw_cost(costs):
    """Exact warp=1, s=1, unconstrained DTW; accumulate in float64.

    Cells on the same anti-diagonal are independent. All costs and the DP
    matrix remain on the input device; no per-cell Python or CPU transfer.
    """
    if costs.ndim != 2 or min(costs.shape) == 0:
        raise ValueError("Expected a nonempty pairwise cost matrix.")
    rows, cols = costs.shape
    dp = torch.full(
        (rows + 1, cols + 1), float("inf"),
        dtype=torch.float64, device=costs.device,
    )
    dp[0, 0] = 0
    costs = costs.double()
    for diagonal in range(rows + cols - 1):
        i = torch.arange(
            max(0, diagonal - cols + 1), min(rows - 1, diagonal) + 1,
            device=costs.device,
        )
        j = diagonal - i
        previous = torch.minimum(
            dp[i, j], torch.minimum(dp[i, j + 1], dp[i + 1, j])
        )
        dp[i + 1, j + 1] = costs[i, j] + previous
    return dp[-1, -1]


@torch.no_grad()
def dtw_pa_jpe(x, y, wanted=None, pair_chunk_size=4096):
    """Return the legacy cumulative DTW-PA-JPE, evaluated on x.device."""
    return dtw_cost(pairwise_pa_jpe(x, y, wanted, pair_chunk_size))
