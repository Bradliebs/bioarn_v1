"""BIO_ARN_DTP: Difference Target Propagation contrastive encoder."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bioarn.core.simclr import SimCLRProjectionHead, nt_xent_loss

__all__ = ["DTPTrainMetrics", "DTPContrastiveEncoder"]


class DTPTrainMetrics(NamedTuple):
    contrastive_loss: float
    layer_losses: tuple[float, float, float]
    inv_losses: tuple[float, float]
    target_displacements: tuple[float, float, float]


class DTPContrastiveEncoder(nn.Module):
    """
    BIO_ARN_DTP: encoder trained via Difference Target Propagation.

    Autograd boundaries:
    - BACKPROP_PROJECTION: projection head uses full autograd
    - BACKPROP_INVERSE: inverse models use reconstruction-loss autograd
    - Encoder layers: local target MSE autograd ONLY within each layer
    """

    CHANNELS = (96, 384, 512)
    KERNEL_SIZES = (5, 3, 3)

    def __init__(
        self,
        temperature: float = 0.5,
        step_size: float = 0.3,
        proj_lr: float = 3e-4,
        encoder_lr: float = 1e-4,
        inverse_lr: float = 3e-4,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.step_size = float(step_size)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.enc_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2, bias=False),
                    nn.BatchNorm2d(96),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ),
                nn.Sequential(
                    nn.Conv2d(96, 384, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(384),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ),
                nn.Sequential(
                    nn.Conv2d(384, 512, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        self.proj_head = SimCLRProjectionHead(in_dim=self.output_dim, hidden_dim=self.output_dim, out_dim=128)
        self.inv_models = nn.ModuleList(
            [
                self._make_inverse_model(384, 96),
                self._make_inverse_model(512, 384),
            ]
        )

        self.proj_opt = torch.optim.Adam(self.proj_head.parameters(), lr=proj_lr)
        self.enc_opts = [
            torch.optim.Adam(layer.parameters(), lr=encoder_lr)
            for layer in self.enc_layers
        ]
        self.inv_opts = [
            torch.optim.Adam(inv_model.parameters(), lr=inverse_lr)
            for inv_model in self.inv_models
        ]

    @staticmethod
    def _make_inverse_model(in_dim: int, out_dim: int) -> nn.Sequential:
        hidden_dim = max(int(in_dim), int(out_dim))
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    @property
    def output_dim(self) -> int:
        return int(self.CHANNELS[-1])

    def _pool_flat(self, x: Tensor) -> Tensor:
        return self.gap(x).flatten(1)

    @staticmethod
    def _mean_displacement(target_a: Tensor, source_a: Tensor, target_b: Tensor, source_b: Tensor) -> Tensor:
        disp_a = (target_a - source_a).norm(dim=1).mean()
        disp_b = (target_b - source_b).norm(dim=1).mean()
        return 0.5 * (disp_a + disp_b)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        x = self.enc_layers[0](x)
        x = self.enc_layers[1](x)
        x = self.enc_layers[2](x)
        return self._pool_flat(x)

    def train_step(self, view_a: Tensor, view_b: Tensor) -> DTPTrainMetrics:
        if view_a.shape != view_b.shape:
            raise ValueError(f"Expected matching view shapes, got {tuple(view_a.shape)} and {tuple(view_b.shape)}")
        if view_a.ndim != 4:
            raise ValueError(f"Expected rank-4 image batches, got shape {tuple(view_a.shape)}")

        self.train()

        with torch.no_grad():
            sp1a = self.enc_layers[0](view_a)
            sp2a = self.enc_layers[1](sp1a)
            h3a = self._pool_flat(self.enc_layers[2](sp2a))
            h2a = self._pool_flat(sp2a)
            h1a = self._pool_flat(sp1a)

            sp1b = self.enc_layers[0](view_b)
            sp2b = self.enc_layers[1](sp1b)
            h3b = self._pool_flat(self.enc_layers[2](sp2b))
            h2b = self._pool_flat(sp2b)
            h1b = self._pool_flat(sp1b)

        h3a_g = h3a.detach().requires_grad_(True)
        h3b_g = h3b.detach().requires_grad_(True)
        self.proj_opt.zero_grad(set_to_none=True)
        z_a = F.normalize(self.proj_head(h3a_g), dim=1)
        z_b = F.normalize(self.proj_head(h3b_g), dim=1)
        contrastive_loss = nt_xent_loss(z_a, z_b, temperature=self.temperature)
        contrastive_loss.backward()
        grad_h3a = h3a_g.grad
        grad_h3b = h3b_g.grad
        if grad_h3a is None or grad_h3b is None:
            raise RuntimeError("Projection head failed to produce top-layer gradients")
        self.proj_opt.step()

        with torch.no_grad():
            t3a = h3a - self.step_size * F.normalize(grad_h3a, dim=1, eps=1e-8)
            t3b = h3b - self.step_size * F.normalize(grad_h3b, dim=1, eps=1e-8)

            t2a = self.inv_models[1](t3a) + (h2a - self.inv_models[1](h3a))
            t2b = self.inv_models[1](t3b) + (h2b - self.inv_models[1](h3b))
            t1a = self.inv_models[0](t2a) + (h1a - self.inv_models[0](h2a))
            t1b = self.inv_models[0](t2b) + (h1b - self.inv_models[0](h2b))

            disp3 = self._mean_displacement(t3a, h3a, t3b, h3b)
            disp2 = self._mean_displacement(t2a, h2a, t2b, h2b)
            disp1 = self._mean_displacement(t1a, h1a, t1b, h1b)

        self.enc_opts[2].zero_grad(set_to_none=True)
        h3_pred_a = self._pool_flat(self.enc_layers[2](sp2a.detach()))
        h3_pred_b = self._pool_flat(self.enc_layers[2](sp2b.detach()))
        loss3 = F.mse_loss(h3_pred_a, t3a.detach()) + F.mse_loss(h3_pred_b, t3b.detach())
        loss3.backward()
        self.enc_opts[2].step()

        self.enc_opts[1].zero_grad(set_to_none=True)
        h2_pred_a = self._pool_flat(self.enc_layers[1](sp1a.detach()))
        h2_pred_b = self._pool_flat(self.enc_layers[1](sp1b.detach()))
        loss2 = F.mse_loss(h2_pred_a, t2a.detach()) + F.mse_loss(h2_pred_b, t2b.detach())
        loss2.backward()
        self.enc_opts[1].step()

        self.enc_opts[0].zero_grad(set_to_none=True)
        h1_pred_a = self._pool_flat(self.enc_layers[0](view_a.detach()))
        h1_pred_b = self._pool_flat(self.enc_layers[0](view_b.detach()))
        loss1 = F.mse_loss(h1_pred_a, t1a.detach()) + F.mse_loss(h1_pred_b, t1b.detach())
        loss1.backward()
        self.enc_opts[0].step()

        self.inv_opts[0].zero_grad(set_to_none=True)
        inv_loss0 = F.mse_loss(self.inv_models[0](h2a.detach()), h1a.detach()) + F.mse_loss(
            self.inv_models[0](h2b.detach()), h1b.detach()
        )
        inv_loss0.backward()
        self.inv_opts[0].step()

        self.inv_opts[1].zero_grad(set_to_none=True)
        inv_loss1 = F.mse_loss(self.inv_models[1](h3a.detach()), h2a.detach()) + F.mse_loss(
            self.inv_models[1](h3b.detach()), h2b.detach()
        )
        inv_loss1.backward()
        self.inv_opts[1].step()

        return DTPTrainMetrics(
            contrastive_loss=float(contrastive_loss.item()),
            layer_losses=(float(loss3.item()), float(loss2.item()), float(loss1.item())),
            inv_losses=(float(inv_loss0.item()), float(inv_loss1.item())),
            target_displacements=(float(disp3.item()), float(disp2.item()), float(disp1.item())),
        )
