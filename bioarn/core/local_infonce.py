"""Lane 2 — Local InfoNCE approximation. No backprop through encoder. Bio-ARN-compatible."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bioarn.core.softhebb_net import SoftHebbNet

__all__ = ["LocalInfoNCEEncoder"]


class LocalInfoNCEEncoder(nn.Module):
    def __init__(
        self,
        channels: tuple[int, ...] = (96, 384, 512),
        kernel_sizes: tuple[int, ...] = (5, 3, 3),
        gamma: float = 2.0,
        eta: float = 0.01,
        global_pool: bool = True,
        oja_decay: float = 0.05,
        filter_decorr: float = 0.02,
        neg_weight: float = 0.5,
        temperature: float = 0.5,
    ) -> None:
        super().__init__()
        self.oja_decay = float(oja_decay)
        self.filter_decorr = float(filter_decorr)
        self.neg_weight = float(neg_weight)
        self.temperature = float(temperature)
        self._last_pos_sim = 0.0
        self._last_neg_sim = 0.0
        try:
            self._supports_oja = True
            self.encoder = SoftHebbNet(
                channels=channels,
                kernel_sizes=kernel_sizes,
                gamma=gamma,
                eta=eta,
                global_pool=global_pool,
                oja_decay=oja_decay,
                filter_decorr=filter_decorr,
            )
        except TypeError:
            self._supports_oja = False
            self.encoder = SoftHebbNet(
                channels=channels,
                kernel_sizes=kernel_sizes,
                gamma=gamma,
                eta=eta,
                global_pool=global_pool,
            )
        self._fallback_oja_sums: list[Tensor] = []
        self._fallback_oja_counts: list[Tensor] = []

    @property
    def output_dim(self) -> int:
        return self.encoder.output_dim

    def _positive_modulation(self, similarity: Tensor) -> Tensor:
        return ((1.0 + similarity) / 2.0).clamp(0.0, 1.0)

    def _negative_modulation(self, similarity: Tensor) -> Tensor:
        return (similarity / max(self.temperature, 1e-6)).clamp(0.0, 1.0)

    def _ensure_fallback_buffers(self) -> None:
        if self._supports_oja:
            return
        layers = list(self.encoder.layers)
        if len(self._fallback_oja_sums) != len(layers):
            self._fallback_oja_sums = []
            self._fallback_oja_counts = []
        for idx, layer in enumerate(layers):
            need_init = idx >= len(self._fallback_oja_sums)
            if not need_init:
                need_init = (
                    self._fallback_oja_sums[idx].device != layer.weight.device
                    or self._fallback_oja_sums[idx].dtype != layer.weight.dtype
                )
            if need_init:
                if idx < len(self._fallback_oja_sums):
                    self._fallback_oja_sums[idx] = torch.zeros(layer.weight.shape[0], device=layer.weight.device, dtype=layer.weight.dtype)
                    self._fallback_oja_counts[idx] = torch.zeros(1, device=layer.weight.device, dtype=layer.weight.dtype)
                else:
                    self._fallback_oja_sums.append(
                        torch.zeros(layer.weight.shape[0], device=layer.weight.device, dtype=layer.weight.dtype)
                    )
                    self._fallback_oja_counts.append(torch.zeros(1, device=layer.weight.device, dtype=layer.weight.dtype))

    @staticmethod
    def _clamp_filters(weight: Tensor, decorrelation: float) -> None:
        flat = weight.view(weight.shape[0], -1)
        flat.sub_(flat.mean(dim=1, keepdim=True))
        if decorrelation > 0.0 and flat.shape[0] > 1:
            normalized = F.normalize(flat, dim=1, eps=1e-12)
            similarity = normalized @ normalized.T
            similarity.fill_diagonal_(0.0)
            flat.sub_(float(decorrelation) * (similarity @ flat) / max(flat.shape[0] - 1, 1))
        flat.div_(flat.norm(dim=1, keepdim=True).clamp_min(1e-6))

    @torch.no_grad()
    def _accumulate_signed_update(self, x: Tensor, signal: Tensor) -> None:
        self._ensure_fallback_buffers()
        current = x
        sample_signal = signal.to(device=x.device, dtype=x.dtype).reshape(-1, 1, 1, 1)
        for idx, layer in enumerate(self.encoder.layers):
            pre = F.conv2d(current, layer._w_hat, padding=layer.padding)
            activ = F.relu(pre)
            winners = F.softmax(layer.gamma * activ, dim=1)
            weighted = winners * sample_signal
            layer._accumulate(current.detach(), weighted.detach())
            if not self._supports_oja:
                self._fallback_oja_sums[idx].add_(weighted.square().sum(dim=(0, 2, 3)).to(layer.weight.dtype))
                self._fallback_oja_counts[idx].add_(float(weighted.shape[0] * weighted.shape[2] * weighted.shape[3]))
            current = winners * activ
            if idx < len(self.encoder.layers) - 1:
                current = F.max_pool2d(current, kernel_size=2, stride=2)

    @torch.no_grad()
    def hebbian_update(self, view_a: Tensor, view_b: Tensor) -> None:
        self.encoder.eval()
        feat_a = self.encoder(view_a)
        feat_b = self.encoder(view_b)

        feat_a_n = F.normalize(feat_a, dim=1)
        feat_b_n = F.normalize(feat_b, dim=1)
        pos_sim = (feat_a_n * feat_b_n).sum(dim=1)
        pos_mod = self._positive_modulation(pos_sim)

        neg_idx = torch.randperm(view_b.shape[0], device=view_b.device)
        if view_b.shape[0] > 1 and torch.equal(neg_idx, torch.arange(view_b.shape[0], device=view_b.device)):
            neg_idx = neg_idx.roll(1)
        feat_b_neg = self.encoder(view_b[neg_idx])
        feat_b_neg_n = F.normalize(feat_b_neg, dim=1)
        neg_sim = (feat_a_n * feat_b_neg_n).sum(dim=1)
        neg_mod = self._negative_modulation(neg_sim)

        self._accumulate_signed_update(view_a, pos_mod)
        self._accumulate_signed_update(view_b, pos_mod)
        self._accumulate_signed_update(view_a, -self.neg_weight * neg_mod)
        self._accumulate_signed_update(view_b[neg_idx], -self.neg_weight * neg_mod)

        self._last_pos_sim = float(pos_mod.mean().item())
        self._last_neg_sim = float(neg_mod.mean().item())

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def flush_hebbian_updates(self) -> None:
        if self._supports_oja:
            self.encoder.flush_hebbian_updates()
            return

        self._ensure_fallback_buffers()
        for idx, layer in enumerate(self.encoder.layers):
            n = int(layer._n_acc.item())
            if n == 0:
                continue
            dw = (layer._dw_acc / n).reshape_as(layer.weight.data)
            if self.oja_decay > 0.0 and float(self._fallback_oja_counts[idx].item()) > 0.0:
                oja_drive = self._fallback_oja_sums[idx] / self._fallback_oja_counts[idx].clamp_min(1.0)
                flat_weight = layer.weight.data.view(layer.weight.shape[0], -1)
                dw = (
                    dw.view(layer.weight.shape[0], -1)
                    - (self.oja_decay * oja_drive.unsqueeze(1) * flat_weight)
                ).reshape_as(layer.weight.data)
            layer.weight.data.add_(dw, alpha=layer.eta)
            self._clamp_filters(layer.weight.data, decorrelation=self.filter_decorr)
            layer._dw_acc.zero_()
            layer._n_acc.zero_()
            self._fallback_oja_sums[idx].zero_()
            self._fallback_oja_counts[idx].zero_()

    def to(self, *args, **kwargs):
        self.encoder.to(*args, **kwargs)
        self._fallback_oja_sums = []
        self._fallback_oja_counts = []
        return self

    def eval(self):
        self.encoder.eval()
        return self

    def train(self, mode: bool = True):
        self.encoder.train(mode)
        return self
