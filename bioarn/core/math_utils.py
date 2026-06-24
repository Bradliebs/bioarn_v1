"""Mathematical utilities for Bio-ARN: cosine similarity, normalization, sparse ops."""

import torch
import torch.nn.functional as F


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute cosine similarity between vectors or batches of vectors.

    Args:
        a: Tensor of shape (..., D).
        b: Tensor of shape (..., D).
        eps: Small value to avoid division by zero.

    Returns:
        Cosine similarity, same batch shape as inputs.
    """
    a_norm = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a_norm * b_norm).sum(dim=-1)


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize vectors to unit length along the last dimension."""
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def sparse_top_k(x: torch.Tensor, k: int) -> torch.Tensor:
    """Keep only the top-k values per row, zero out the rest.

    Implements competitive inhibition: only the strongest activations survive.

    Args:
        x: Tensor of shape (batch, features) or (features,).
        k: Number of values to keep.

    Returns:
        Tensor with same shape, non-top-k values zeroed.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    topk_vals, topk_idx = torch.topk(x, k=min(k, x.shape[-1]), dim=-1)
    result = torch.zeros_like(x)
    result.scatter_(-1, topk_idx, topk_vals)

    if squeeze:
        result = result.squeeze(0)
    return result


def hamming_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute Hamming distance between binary vectors.

    Args:
        a: Binary tensor of shape (..., D).
        b: Binary tensor of shape (..., D).

    Returns:
        Hamming distance (number of differing bits).
    """
    return (a != b).sum(dim=-1)


def to_binary(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert continuous values to binary by thresholding."""
    return (x > threshold).float()


def spike_rate(spikes: torch.Tensor, window: int) -> torch.Tensor:
    """Compute firing rate from a spike train over a time window.

    Args:
        spikes: Binary tensor of shape (time, ...).
        window: Number of time steps to average over.

    Returns:
        Firing rate, shape (...,).
    """
    if spikes.shape[0] < window:
        window = spikes.shape[0]
    return spikes[-window:].mean(dim=0)
