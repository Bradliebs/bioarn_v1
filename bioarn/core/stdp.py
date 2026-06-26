"""Spike-timing-dependent plasticity utilities for CCC feedback learning."""

from __future__ import annotations

import math

import torch
from torch import nn

from bioarn.config import STDPConfig
from bioarn.core.math_utils import normalize


class STDPRule(nn.Module):
    """Pair-based STDP with exponentially decaying pre/post traces."""

    def __init__(self, config: STDPConfig, *, num_pre: int, num_post: int) -> None:
        super().__init__()
        self.config = config
        self.num_pre = int(num_pre)
        self.num_post = int(num_post)
        self.register_buffer("pre_trace", torch.zeros(self.num_pre, dtype=torch.float32))
        self.register_buffer("post_trace", torch.zeros(self.num_post, dtype=torch.float32))
        self.register_buffer("last_timestep", torch.tensor(-1, dtype=torch.long))

    @staticmethod
    def _prepare_pre(pre_spikes: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        return (pre_spikes.reshape(-1) > 0).to(dtype)

    @staticmethod
    def _prepare_post(post_activity: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        post = post_activity.reshape(-1).to(dtype).clamp_min(0.0)
        if float(post.norm().item()) <= 1e-8:
            return torch.zeros_like(post)
        return normalize(post.unsqueeze(0)).squeeze(0)

    def _advance(self, timestep: int) -> None:
        current = int(timestep)
        previous = int(self.last_timestep.item())
        if previous >= 0:
            delta_t = max(0, current - previous)
            if delta_t > 0:
                self.pre_trace.mul_(math.exp(-delta_t / max(self.config.tau_plus, 1e-6)))
                self.post_trace.mul_(math.exp(-delta_t / max(self.config.tau_minus, 1e-6)))
        self.last_timestep.fill_(current)

    @torch.no_grad()
    def reset_state(self) -> None:
        self.pre_trace.zero_()
        self.post_trace.zero_()
        self.last_timestep.fill_(-1)

    @torch.no_grad()
    def observe_pre_spikes(self, pre_spikes: torch.Tensor, *, timestep: int) -> None:
        self._advance(timestep)
        self.pre_trace.add_(self._prepare_pre(pre_spikes, dtype=self.pre_trace.dtype))

    @torch.no_grad()
    def observe_post_spike(self, post_activity: torch.Tensor, *, timestep: int) -> None:
        self._advance(timestep)
        self.post_trace.add_(self._prepare_post(post_activity, dtype=self.post_trace.dtype))

    @torch.no_grad()
    def step(
        self,
        pre_spikes: torch.Tensor,
        *,
        post_spike: bool,
        post_activity: torch.Tensor,
        timestep: int,
    ) -> torch.Tensor:
        """Return an STDP weight delta for the current pre/post events."""

        self._advance(timestep)
        pre_event = self._prepare_pre(pre_spikes, dtype=self.pre_trace.dtype)
        post_event = (
            self._prepare_post(post_activity, dtype=self.post_trace.dtype)
            if bool(post_spike)
            else torch.zeros_like(self.post_trace)
        )

        update = torch.zeros(
            self.num_pre,
            self.num_post,
            device=self.pre_trace.device,
            dtype=self.pre_trace.dtype,
        )
        if bool(pre_event.any().item()) and bool(self.post_trace.any().item()):
            update.sub_(
                self.config.A_minus
                * pre_event.unsqueeze(-1)
                * self.post_trace.unsqueeze(0)
            )
        if bool(post_event.any().item()) and bool(self.pre_trace.any().item()):
            update.add_(
                self.config.A_plus
                * self.pre_trace.unsqueeze(-1)
                * post_event.unsqueeze(0)
            )

        self.pre_trace.add_(pre_event)
        self.post_trace.add_(post_event)
        return update


__all__ = ["STDPRule"]
