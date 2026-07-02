"""Lane 1 — Backprop InfoNCE upper bound. BACKPROP_UPPER_BOUND: not Bio-ARN-pure."""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "SimCLRBackbone",
    "SimCLRProjectionHead",
    "SimCLRNet",
    "nt_xent_loss",
    "simclr_augment",
]



def simclr_augment(batch: Tensor, rng: random.Random) -> Tensor:
    """Batched SimCLR-style augmentation. All B images processed in parallel on device."""
    B, C, H, W = batch.shape
    device = batch.device

    # 1. Random resized crop — batched via affine_grid + grid_sample (one CUDA call)
    thetas: list[list[list[float]]] = []
    for _ in range(B):
        s = rng.uniform(0.2, 1.0)
        r = math.exp(rng.uniform(math.log(0.75), math.log(1.33)))
        ch = max(1, min(int(round(math.sqrt(s * H * W / r))), H))
        cw = max(1, min(int(round(math.sqrt(s * H * W * r))), W))
        top = rng.randint(0, H - ch) if ch < H else 0
        left = rng.randint(0, W - cw) if cw < W else 0
        sx = cw / W
        sy = ch / H
        tx = (2.0 * left + cw) / W - 1.0
        ty = (2.0 * top + ch) / H - 1.0
        thetas.append([[sx, 0.0, tx], [0.0, sy, ty]])
    theta_t = torch.tensor(thetas, dtype=torch.float32, device=device)
    grid = F.affine_grid(theta_t, [B, C, H, W], align_corners=False)
    result = F.grid_sample(batch.float(), grid, mode="bilinear", align_corners=False, padding_mode="reflection")

    # 2. Random horizontal flip — vectorized
    flip = torch.tensor([rng.random() < 0.5 for _ in range(B)], dtype=torch.bool, device=device)
    if flip.any():
        result[flip] = result[flip].flip(-1)

    # 3. Color jitter (p=0.8) — RGB-only ops, no HSV (avoids per-image Python overhead)
    for i in range(B):
        if rng.random() < 0.8:
            img = result[i]
            ops = ["brightness", "contrast", "saturation"]
            rng.shuffle(ops)
            for op in ops:
                if op == "brightness":
                    img = (img * rng.uniform(0.6, 1.4)).clamp(0.0, 1.0)
                elif op == "contrast":
                    mean = img.mean(dim=(1, 2), keepdim=True)
                    img = ((img - mean) * rng.uniform(0.6, 1.4) + mean).clamp(0.0, 1.0)
                else:
                    gray = 0.2989 * img[0:1] + 0.5870 * img[1:2] + 0.1140 * img[2:3]
                    img = ((img - gray) * rng.uniform(0.6, 1.4) + gray).clamp(0.0, 1.0)
            result[i] = img

    # 4. Random grayscale (p=0.2) — vectorized
    for i in range(B):
        if rng.random() < 0.2:
            gray = 0.2989 * result[i, 0:1] + 0.5870 * result[i, 1:2] + 0.1140 * result[i, 2:3]
            result[i] = gray.expand(C, H, W)

    return result.to(dtype=batch.dtype)


class SimCLRBackbone(nn.Module):
    def __init__(
        self,
        channels: tuple[int, int, int] = (96, 384, 512),
        kernel_sizes: tuple[int, int, int] = (5, 3, 3),
    ) -> None:
        super().__init__()
        in_channels = (3, *channels[:-1])
        self.conv_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
                for in_ch, out_ch, kernel_size in zip(in_channels, channels, kernel_sizes, strict=True)
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self._output_dim = int(channels[-1])

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: Tensor) -> Tensor:
        for idx, block in enumerate(self.conv_blocks):
            x = block(x)
            if idx < len(self.conv_blocks) - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2)
        return self.pool(x).flatten(1)


class SimCLRProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def nt_xent_loss(z_a: Tensor, z_b: Tensor, temperature: float) -> Tensor:
    """Compute the NT-Xent loss over a batch of positive pairs."""
    if z_a.shape != z_b.shape:
        raise ValueError(f"Expected matching shapes, got {tuple(z_a.shape)} and {tuple(z_b.shape)}")
    if z_a.ndim != 2:
        raise ValueError(f"Expected rank-2 projections, got shape {tuple(z_a.shape)}")

    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)
    n = z_a.shape[0]
    z = torch.cat([z_a, z_b], dim=0)
    logits = (z @ z.T) / max(float(temperature), 1e-6)
    logits = logits.masked_fill(torch.eye(2 * n, device=z.device, dtype=torch.bool), float("-inf"))
    targets = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    return F.cross_entropy(logits, targets)


class SimCLRNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = SimCLRBackbone()
        self.projection_head = SimCLRProjectionHead(in_dim=self.backbone.output_dim)

    @property
    def output_dim(self) -> int:
        return self.backbone.output_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def project(self, x: Tensor) -> Tensor:
        features = self.backbone(x) if x.ndim == 4 else x
        return self.projection_head(features)

    def train_step(
        self,
        view_a: Tensor,
        view_b: Tensor,
        optimizer: torch.optim.Optimizer,
        temperature: float = 0.5,
    ) -> float:
        self.train()
        optimizer.zero_grad(set_to_none=True)
        z_a = F.normalize(self.project(view_a), dim=1)
        z_b = F.normalize(self.project(view_b), dim=1)
        loss = nt_xent_loss(z_a, z_b, temperature=temperature)
        loss.backward()
        optimizer.step()
        return float(loss.item())
