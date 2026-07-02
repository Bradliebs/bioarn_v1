"""SoftHebb Convolutional Network — Journé et al. (ICLR 2023) variant.

A clean, standalone implementation of multilayer Hebbian feature learning using
Soft Winner-Take-All (SoftWTA) competition.  Distinct from ConvF1Layer which uses
hard top-K gating — SoftWTA uses a differentiable softmax over channel activations.

Key differences from ConvF1Layer's ``softhebb_enabled`` flag:
- Filters are L2-normalised BEFORE computing activations (not after)
- Update: ΔW ∝ g ⊗ x_patches  (g = SoftWTA weight = softmax(γ·a, dim_channel))
- Explicit per-filter weight renormalisation after each update batch
- MaxPool2d(2×2) between layers (not adaptive spatial grid)
- No Oja decay mixing, no spatial_top_k, no filter_decorrelation — just WTA + update

Architecture (default, compact Journé variant):
    Input [B, 3, 32, 32]
    → SoftHebbLayer(96, k=5, pad=2)  → ReLU → SoftWTA  → MaxPool2
    → SoftHebbLayer(384, k=3, pad=1) → ReLU → SoftWTA  → MaxPool2
    → SoftHebbLayer(512, k=3, pad=1) → ReLU → SoftWTA
    → [global_pool=True]  AdaptiveAvgPool(1×1) → flatten [B, 512]
    → [global_pool=False] AdaptiveAvgPool(4×4) → flatten [B, 512×4×4 = 8192]

Phase 3b default is global_pool=True (512-dim).  The spatial 8192-dim
variant (global_pool=False) is kept as a diagnostic comparison only.

Interface mirrors ConvF1Layer so it plugs into the existing eval pipeline:
    model(x)                           → dense [B, output_dim] for feature extraction
    model.hebbian_update(x, signal)    → accumulate Hebbian updates (training mode)
    model.flush_hebbian_updates()      → apply accumulated updates
    model.output_dim                   → int
"""

from __future__ import annotations

import math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["SoftHebbLayer", "SoftHebbNet"]

# ─── Single convolutional SoftHebb layer ──────────────────────────────────────

class SoftHebbLayer(nn.Module):
    """Conv layer trained by SoftWTA Hebbian updates.

    Forward pass:
        1. Normalise weight rows (per filter) → w_hat
        2. u = conv(x, w_hat);  a = relu(u)
        3. g = softmax(γ · a, dim=1)   — SoftWTA competition
        4. output = g · a              — modulated activation

    Hebbian accumulation (training mode only):
        dW[c, :] += Σ_{b,h,w} g[b,c,h,w] · x_patch[b,h,w]

    Weight update (flush):
        W ← W + η · dW_normalised
        W ← W / ‖W‖_2_per_filter        — renormalise
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        *,
        gamma: float = 10.0,
        eta: float = 0.01,
        oja_decay: float = 0.0,
        filter_decorr: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size) | 1  # force odd
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.oja_decay = float(oja_decay)
        self.filter_decorr = float(filter_decorr)
        self.padding = self.kernel_size // 2

        weight = torch.empty(out_channels, in_channels, self.kernel_size, self.kernel_size)
        nn.init.kaiming_normal_(weight, nonlinearity="relu")
        self.weight = nn.Parameter(weight, requires_grad=False)
        self._normalise_weights_()

        patch_dim = in_channels * self.kernel_size * self.kernel_size
        self.register_buffer("_dw_acc", torch.zeros(out_channels, patch_dim))
        self.register_buffer("_n_acc", torch.tensor(0, dtype=torch.int64))

    # ── Properties ────────────────────────────────────────────────────────────

    def _normalise_weights_(self) -> None:
        """In-place per-filter L2 normalisation."""
        with torch.no_grad():
            w = self.weight.data
            norms = w.reshape(w.shape[0], -1).norm(dim=1).reshape(-1, 1, 1, 1).clamp(min=1e-6)
            self.weight.data = w / norms

    @property
    def _w_hat(self) -> torch.Tensor:
        """Normalised weight (no in-place update needed — already normalised)."""
        return self.weight.data

    # ── Forward ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Accumulates Hebbian update when self.training is True."""
        u = F.conv2d(x, self._w_hat, padding=self.padding)
        a = F.relu(u)
        g = F.softmax(self.gamma * a, dim=1)
        if self.training:
            self._accumulate(x.detach(), g.detach())
        return g * a

    # ── Hebbian accumulation ──────────────────────────────────────────────────

    def _accumulate(self, x: torch.Tensor, g: torch.Tensor) -> None:
        """Accumulate correlation between SoftWTA response and input patches."""
        B, _, H, W = g.shape
        # Unfold input into patches: [B, in_ch*kH*kW, H*W]
        patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding)
        # Reshape g to [B, out_ch, H*W]
        g_flat = g.reshape(B, self.out_channels, H * W)
        # Correlation: dW[c, p] = Σ_b Σ_hw  g[b,c,hw] * patches[b,p,hw]
        dw = torch.einsum("bch,bph->cp", g_flat, patches)
        self._dw_acc.add_(dw)
        self._n_acc.add_(B * H * W)

    def flush(self) -> None:
        """Apply accumulated Hebbian update, Oja decay, filter decorrelation, then renormalise."""
        n = int(self._n_acc.item())
        if n == 0:
            return
        dw = (self._dw_acc / n).reshape_as(self.weight.data)
        self.weight.data.add_(dw, alpha=self.eta)

        if self.oja_decay > 0.0:
            # Oja decay: prevent weight magnitude explosion, maintain unit-norm tendency
            w_flat = self.weight.data.reshape(self.out_channels, -1)
            w_norm_sq = (w_flat ** 2).sum(dim=1, keepdim=True)
            self.weight.data.sub_(
                (self.oja_decay * self.weight.data.reshape(self.out_channels, -1) * w_norm_sq)
                .reshape_as(self.weight.data)
            )

        if self.filter_decorr > 0.0:
            # Filter decorrelation: push filter pair-wise correlations toward zero
            w_flat = self.weight.data.reshape(self.out_channels, -1)
            w_norm = F.normalize(w_flat, dim=1, eps=1e-8)
            gram = w_norm @ w_norm.T  # [C, C] pairwise cosine similarities
            gram.fill_diagonal_(0.0)
            decorr_signal = (gram @ w_norm).reshape_as(self.weight.data)
            self.weight.data.sub_(decorr_signal, alpha=self.filter_decorr)

        self._normalise_weights_()
        self._dw_acc.zero_()
        self._n_acc.zero_()


# ─── Multi-layer network ───────────────────────────────────────────────────────

class SoftHebbNet(nn.Module):
    """Multi-layer SoftHebb CNN.

    Default architecture (compact Journé variant):
        96ch k=5 → MaxPool → 384ch k=3 → MaxPool → 512ch k=3 → AvgPool(4×4)
        output_dim = 512 × 4 × 4 = 8192

    Plugs into the existing evaluation pipeline that expects:
        model(x)                        → [B, output_dim] dense features
        model.hebbian_update(x, sig)    → accumulate updates (training)
        model.flush_hebbian_updates()   → apply updates
        model.output_dim                → int
    """

    SPATIAL_GRID: int = 4

    def __init__(
        self,
        channels: tuple[int, ...] = (96, 384, 512),
        kernel_sizes: tuple[int, ...] = (5, 3, 3),
        *,
        gamma: float = 10.0,
        eta: float = 0.01,
        global_pool: bool = True,
        oja_decay: float = 0.0,
        filter_decorr: float = 0.0,
    ) -> None:
        super().__init__()
        assert len(channels) == len(kernel_sizes), "channels and kernel_sizes must be same length"
        in_chs = [3, *list(channels[:-1])]
        self.layers: nn.ModuleList = nn.ModuleList(
            [
                SoftHebbLayer(
                    in_chs[i], channels[i], kernel_sizes[i],
                    gamma=gamma, eta=eta,
                    oja_decay=oja_decay, filter_decorr=filter_decorr,
                )
                for i in range(len(channels))
            ]
        )
        self._output_channels = int(channels[-1])
        self._global_pool = bool(global_pool)

    @property
    def output_dim(self) -> int:
        if self._global_pool:
            return self._output_channels
        return self._output_channels * self.SPATIAL_GRID * self.SPATIAL_GRID

    @property
    def gamma(self) -> float:
        """Return gamma of the first layer (all layers share the same γ)."""
        return float(self.layers[0].gamma)

    # ── Forward (eval / feature extraction) ──────────────────────────────────

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns dense feature vector [B, output_dim]."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2)
        if self._global_pool:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        else:
            x = F.adaptive_avg_pool2d(x, (self.SPATIAL_GRID, self.SPATIAL_GRID))
        return x.flatten(1)

    # ── Hebbian training ──────────────────────────────────────────────────────

    @torch.no_grad()
    def hebbian_update(
        self,
        x: torch.Tensor,
        learning_signal: torch.Tensor | None = None,
    ) -> None:
        """Run a training-mode forward pass to accumulate Hebbian updates.

        ``learning_signal`` (per-sample scalar) scales each sample's contribution.
        Ignored if None (all samples weighted equally).
        """
        self.train()
        if learning_signal is not None:
            # Scale input by per-sample signal before accumulation.
            # Negative signals produce anti-Hebbian updates (decorrelation).
            scale = learning_signal.reshape(-1, 1, 1, 1)
            x = x * scale
        self(x)

    def flush_hebbian_updates(self) -> None:
        """Apply all accumulated Hebbian updates to every layer."""
        for layer in self.layers:
            layer.flush()
