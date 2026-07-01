"""Local Contrastive Hebbian Encoder — CLAPP-inspired.

Extends SoftHebbNet with a contrastive modulation signal: the Hebbian update
for each sample is weighted by how consistently the network responds to two
different augmented views of the same image.

Learning principle (CLAPP-style, Illing et al. 2021):
    - Same image, two augmented views → features should align (positive pair)
    - Different images             → no alignment required (implicit negative)
    - The local modulation signal is: cos_sim(f1, f2) ∈ [−1, 1]
    - Update is amplified when the two views produce similar features,
      and reduced (or reversed) when they diverge

Why this helps:
    Pure Hebbian updates reinforce ANY consistently-active pattern.
    Contrastive modulation adds a preference for view-invariant features —
    the network learns structure that survives augmentation, not arbitrary
    co-activation.

Implementation:
    For each training batch:
        1. Generate view1 = augment(x),  view2 = augment(x)
        2. Forward view1 → f1 (dense features, normalised)
        3. Forward view2 → f2 (dense features, normalised)
        4. Compute per-sample cos_sim: s = (f1 · f2) / (‖f1‖ · ‖f2‖)
        5. learning_signal = 0.5 + 0.5 * s    ∈ [0, 1]  (always non-negative)
        6. Run base hebbian_update(view1, learning_signal)

Interface:
    Same as SoftHebbNet:
        model(x)                        → [B, output_dim] features
        model.hebbian_update(x, sig)    → contrastive-modulated update
        model.flush_hebbian_updates()   → apply updates
        model.output_dim                → int
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import nn

from bioarn.core.softhebb_net import SoftHebbNet

__all__ = ["LocalContrastiveEncoder"]


# ─── Augmentation helpers ─────────────────────────────────────────────────────

def _augment_view(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Per-call augmentation: flip + random crop + brightness + contrast jitter."""
    if rng.random() > 0.5:
        x = x.flip(-1)
    pad = 4
    padded = F.pad(x, [pad, pad, pad, pad], mode="reflect")
    i = rng.randint(0, 2 * pad)
    j = rng.randint(0, 2 * pad)
    x = padded[:, :, i : i + 32, j : j + 32]
    brightness = rng.uniform(0.7, 1.3)
    x = (x * brightness).clamp(0.0, 1.0)
    contrast = rng.uniform(0.8, 1.2)
    mean = x.mean(dim=(-1, -2), keepdim=True)
    x = ((x - mean) * contrast + mean).clamp(0.0, 1.0)
    return x


# ─── Encoder ──────────────────────────────────────────────────────────────────

class LocalContrastiveEncoder(nn.Module):
    """SoftHebbNet with contrastive view-consistency modulation.

    Wraps a ``SoftHebbNet`` (or any compatible Hebbian encoder) and overrides
    the Hebbian update to weight each sample by the cosine similarity between
    two augmented views.

    The wrapped encoder's weights are updated via the modulated signal — all
    other behaviour (feature extraction, eval mode) is unchanged.
    """

    def __init__(
        self,
        channels: tuple[int, ...] = (96, 384, 512),
        kernel_sizes: tuple[int, ...] = (5, 3, 3),
        *,
        gamma: float = 10.0,
        eta: float = 0.01,
        contrastive_weight: float = 1.0,
        global_pool: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = SoftHebbNet(channels=channels, kernel_sizes=kernel_sizes, gamma=gamma, eta=eta, global_pool=global_pool)
        self.contrastive_weight = float(contrastive_weight)
        self._rng = random.Random(42)

    @property
    def output_dim(self) -> int:
        return self.encoder.output_dim

    # ── Forward (feature extraction) ──────────────────────────────────────────

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Delegate to base encoder for feature extraction."""
        return self.encoder(x)

    # ── Hebbian training with contrastive modulation ──────────────────────────

    @torch.no_grad()
    def hebbian_update(
        self,
        x: torch.Tensor,
        learning_signal: torch.Tensor | None = None,
    ) -> None:
        """Contrastive-modulated Hebbian update.

        Generates two views of the batch, computes their feature cosine similarity,
        and uses that as the per-sample learning signal.

        Args:
            x: Input images [B, 3, 32, 32] float32 [0, 1].
            learning_signal: Optional base signal (multiplied with cosine sim).
        """
        view1 = _augment_view(x, self._rng)
        view2 = _augment_view(x, self._rng)

        # Extract features for both views (eval mode, no Hebbian accumulation)
        self.encoder.eval()
        f1 = self.encoder(view1)  # [B, D]
        f2 = self.encoder(view2)  # [B, D]

        # Per-sample cosine similarity ∈ [−1, 1]
        f1_norm = F.normalize(f1.float(), dim=1)
        f2_norm = F.normalize(f2.float(), dim=1)
        cos_sim = (f1_norm * f2_norm).sum(dim=1)  # [B]

        # Map to [0, 1]: modulation = 0.5 + 0.5 * cos_sim
        modulation = 0.5 + 0.5 * cos_sim * self.contrastive_weight

        # Apply optional outer learning signal
        if learning_signal is not None:
            modulation = modulation * learning_signal.to(modulation.device)

        # Hebbian update on view1 with contrastive modulation
        self.encoder.hebbian_update(view1, learning_signal=modulation)

    def flush_hebbian_updates(self) -> None:
        self.encoder.flush_hebbian_updates()
