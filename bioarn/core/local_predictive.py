"""Local Predictive Hebbian Encoder — masked patch prediction.

Extends SoftHebbNet with a prediction-error modulation signal: each image is
split into patches, one patch is masked, and the prediction error for the
masked patch modulates the Hebbian update.

Learning principle:
    - Divide 32×32 image into a 4×4 grid of 8×8 patches (16 patches total)
    - Randomly mask one patch (replace with mean colour)
    - Forward the masked image through the encoder → features
    - Use a lightweight linear prediction head to predict the masked patch's
      pixel values from the dense feature vector
    - Prediction error (L2 distance) modulates the Hebbian update:
        * High error  → larger update (much to learn from this example)
        * Low error   → smaller update (already captured this structure)

Why this helps:
    Pure Hebbian learning strengthens any repeatedly active pattern.
    Prediction-error modulation adds a preference for features that capture
    predictive structure — spatial context that helps reconstruct missing
    content.  This is analogous to how prediction error drives plasticity
    in predictive coding models of cortical learning.

Implementation:
    For each training batch:
        1. Select a random patch position p ∈ {0..15}
        2. Record the true patch pixel values: target [B, 3*8*8]
        3. Replace the patch with the image mean: x_masked
        4. Run base hebbian_update(x_masked) → features
        5. Forward prediction head on features → predicted patch
        6. Compute prediction error: e = ‖predicted − target‖₂  (per sample)
        7. Normalise error to [0, 1] and use as learning_signal for a second
           hebbian_update pass on the original (unmasked) image

Note: The prediction head is trained with gradient descent in parallel with
the Hebbian encoder.  This is the only non-Hebbian component.

Interface:
    Same as SoftHebbNet:
        model(x)                        → [B, output_dim] features
        model.hebbian_update(x, sig)    → prediction-modulated Hebbian update
        model.flush_hebbian_updates()   → apply updates
        model.output_dim                → int
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F
from torch import nn

from bioarn.core.softhebb_net import SoftHebbNet

__all__ = ["LocalPredictiveEncoder"]

# ─── Constants ────────────────────────────────────────────────────────────────

_IMG_SIZE = 32
_PATCH_SIZE = 8
_GRID = _IMG_SIZE // _PATCH_SIZE  # 4×4 grid → 16 patches
_PATCH_DIM = 3 * _PATCH_SIZE * _PATCH_SIZE  # 192 pixels per patch


# ─── Patch helpers ────────────────────────────────────────────────────────────

def _extract_patch(x: torch.Tensor, row: int, col: int) -> torch.Tensor:
    """Extract patch at grid position (row, col) → [B, 3, P, P] → flatten to [B, 3*P*P]."""
    r, c = row * _PATCH_SIZE, col * _PATCH_SIZE
    patch = x[:, :, r : r + _PATCH_SIZE, c : c + _PATCH_SIZE]
    return patch.reshape(x.shape[0], -1)


def _mask_patch(x: torch.Tensor, row: int, col: int) -> torch.Tensor:
    """Return a copy of x with the specified patch replaced by per-sample channel mean."""
    x_masked = x.clone()
    r, c = row * _PATCH_SIZE, col * _PATCH_SIZE
    mean_colour = x.mean(dim=(-1, -2), keepdim=True)  # [B, 3, 1, 1]
    x_masked[:, :, r : r + _PATCH_SIZE, c : c + _PATCH_SIZE] = mean_colour
    return x_masked


# ─── Encoder ──────────────────────────────────────────────────────────────────

class LocalPredictiveEncoder(nn.Module):
    """SoftHebbNet with masked-patch prediction-error modulation.

    The prediction head is a small 2-layer MLP trained via gradient descent.
    The Hebbian encoder is trained by the prediction-error-modulated rule.
    """

    def __init__(
        self,
        channels: tuple[int, ...] = (96, 384, 512),
        kernel_sizes: tuple[int, ...] = (5, 3, 3),
        *,
        gamma: float = 10.0,
        eta: float = 0.01,
        pred_lr: float = 1e-3,
        error_scale: float = 2.0,
        global_pool: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = SoftHebbNet(channels=channels, kernel_sizes=kernel_sizes, gamma=gamma, eta=eta, global_pool=global_pool)

        # Lightweight prediction head: features → predicted patch pixels
        feat_dim = self.encoder.output_dim
        self.pred_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, _PATCH_DIM),
        )
        self._pred_optim = torch.optim.Adam(self.pred_head.parameters(), lr=pred_lr)
        self._rng = random.Random(42)
        self.error_scale = float(error_scale)

        # Diagnostic tracking
        self._last_pred_loss: float = float("nan")
        self._pred_loss_history: list[float] = []

    @property
    def output_dim(self) -> int:
        return self.encoder.output_dim

    # ── Forward (feature extraction, eval) ───────────────────────────────────

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    # ── Hebbian training with prediction-error modulation ────────────────────

    def hebbian_update(
        self,
        x: torch.Tensor,
        learning_signal: torch.Tensor | None = None,
    ) -> None:
        """Prediction-error-modulated Hebbian update.

        1. Pick a random patch to mask.
        2. Record the true patch pixels as prediction target.
        3. Forward masked image through encoder, predict masked patch.
        4. Compute prediction error per sample.
        5. Use normalised error as Hebbian learning signal on unmasked image.
        6. Update prediction head via gradient descent on prediction loss.

        Args:
            x: [B, 3, 32, 32] float32 [0, 1].
            learning_signal: Optional base signal (multiplied with error modulation).
        """
        row = self._rng.randint(0, _GRID - 1)
        col = self._rng.randint(0, _GRID - 1)

        target_patch = _extract_patch(x, row, col).detach()  # [B, PATCH_DIM]
        x_masked = _mask_patch(x, row, col)

        # ── Update prediction head ─────────────────────────────────────────
        self.encoder.eval()
        with torch.no_grad():
            feats_masked = self.encoder(x_masked)  # [B, feat_dim]

        self.pred_head.train()
        self._pred_optim.zero_grad(set_to_none=True)
        predicted = self.pred_head(feats_masked.float())  # [B, PATCH_DIM]
        pred_loss = F.mse_loss(predicted, target_patch.float())
        pred_loss.backward()
        self._pred_optim.step()
        self._last_pred_loss = float(pred_loss.detach().item())
        self._pred_loss_history.append(self._last_pred_loss)

        # ── Compute per-sample prediction error ────────────────────────────
        with torch.no_grad():
            errors = (predicted.detach() - target_patch.float()).pow(2).mean(dim=1)  # [B]
            # Normalise to [0, 1]: use sigmoid of scaled error
            modulation = torch.sigmoid(self.error_scale * errors - 1.0)

            if learning_signal is not None:
                modulation = modulation * learning_signal.to(modulation.device)

        # ── Hebbian update on unmasked image with error modulation ─────────
        with torch.no_grad():
            self.encoder.hebbian_update(x, learning_signal=modulation)

    def flush_hebbian_updates(self) -> None:
        self.encoder.flush_hebbian_updates()

    def to(self, *args, **kwargs):  # type: ignore[override]
        result = super().to(*args, **kwargs)
        # Move pred_head optimiser state too
        device = next(self.pred_head.parameters()).device
        for state in self._pred_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        return result
