"""Kanerva-style Sparse Distributed Memory for associative recall."""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
from torch import nn

from bioarn.config import SDMConfig
from bioarn.core.math_utils import hamming_distance, to_binary


class SparseDistributedMemory(nn.Module):
    """Sparse distributed associative memory with fixed hard locations."""

    def __init__(self, config: SDMConfig):
        super().__init__()
        self.config = config
        self.address_dim = config.address_dim
        self.hamming_radius = config.hamming_radius
        self.num_hard_locations = config.num_hard_locations
        self.data_dim = config.data_dim
        self.decay_rate = config.decay_rate

        hard_locations = torch.randint(
            0,
            2,
            (self.num_hard_locations, self.address_dim),
            dtype=torch.float32,
        )
        self.register_buffer("hard_locations", hard_locations)
        self.register_buffer(
            "data_matrix",
            torch.zeros(self.num_hard_locations, self.data_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "activation_counts",
            torch.zeros(self.num_hard_locations, dtype=torch.float32),
        )
        self.register_buffer("address_projection", torch.empty(0, dtype=torch.float32))
        self._projection_input_dim: int | None = None

    def _ensure_2d(self, tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if tensor.dim() == 1:
            return tensor.unsqueeze(0), True
        return tensor, False

    def _get_or_create_projection(
        self,
        input_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self._projection_input_dim is None:
            projection = torch.randn(
                input_dim,
                self.address_dim,
                device=device,
                dtype=dtype,
            ) / math.sqrt(max(input_dim, 1))
            self.address_projection = projection
            self._projection_input_dim = input_dim
        elif self._projection_input_dim != input_dim:
            raise ValueError(
                "SparseDistributedMemory projection was initialized for "
                f"{self._projection_input_dim} input dims, but received {input_dim}."
            )
        return self.address_projection.to(device=device, dtype=dtype)

    def _hamming_to_locations(self, address: torch.Tensor) -> torch.Tensor:
        address = address.to(dtype=self.hard_locations.dtype)
        address_ones = address.sum(dim=-1, keepdim=True)
        location_ones = self.hard_locations.sum(dim=-1).unsqueeze(0)
        overlaps = address @ self.hard_locations.T
        return address_ones + location_ones - 2.0 * overlaps

    def _activated_mask_from_binary(self, address: torch.Tensor) -> torch.Tensor:
        distances = self._hamming_to_locations(address)
        return distances <= float(self.hamming_radius)

    def compute_address(self, concept_direction: torch.Tensor) -> torch.Tensor:
        """Convert a continuous concept direction vector to a binary SDM address."""
        concept_direction, squeeze = self._ensure_2d(concept_direction.to(torch.float32))

        if concept_direction.shape[-1] != self.address_dim:
            projection = self._get_or_create_projection(
                input_dim=concept_direction.shape[-1],
                device=concept_direction.device,
                dtype=concept_direction.dtype,
            )
            concept_direction = concept_direction @ projection

        address = to_binary(concept_direction, threshold=0.0)
        return address.squeeze(0) if squeeze else address

    def get_activated_locations(self, address: torch.Tensor) -> torch.Tensor:
        """Return hard locations within the configured Hamming radius."""
        address = self.compute_address(address)
        address, squeeze = self._ensure_2d(address)
        activated = self._activated_mask_from_binary(address)
        return activated.squeeze(0) if squeeze else activated

    @torch.no_grad()
    def _write_impl(
        self,
        address: torch.Tensor,
        data: torch.Tensor,
        apply_decay: bool = True,
    ) -> None:
        address = self.compute_address(address)
        address, _ = self._ensure_2d(address)
        data, _ = self._ensure_2d(data.to(self.data_matrix.dtype))

        if address.shape[0] != data.shape[0]:
            raise ValueError(
                "Address and data batch sizes must match: "
                f"{address.shape[0]} != {data.shape[0]}"
            )
        if data.shape[-1] != self.data_dim:
            raise ValueError(
                f"Expected data_dim={self.data_dim}, received {data.shape[-1]}."
            )

        activated = self._activated_mask_from_binary(address).to(self.data_matrix.dtype)
        self.data_matrix.add_(activated.T @ data)
        self.activation_counts.add_(activated.sum(dim=0))

        if apply_decay:
            self.data_matrix.mul_(self.decay_rate)

    @torch.no_grad()
    def write(self, address: torch.Tensor, data: torch.Tensor) -> None:
        """Store data at all hard locations activated by the address."""
        self._write_impl(address, data, apply_decay=True)

    def read(self, address: torch.Tensor) -> torch.Tensor:
        """Retrieve data by summing over activated hard locations."""
        address = self.compute_address(address)
        address, squeeze = self._ensure_2d(address)
        activated = self._activated_mask_from_binary(address).to(self.data_matrix.dtype)

        retrieved = activated @ self.data_matrix
        counts = activated @ self.activation_counts
        retrieved = retrieved / counts.clamp_min(1.0).unsqueeze(-1)

        return retrieved.squeeze(0) if squeeze else retrieved

    @torch.no_grad()
    def associate(
        self,
        address_a: torch.Tensor,
        address_b: torch.Tensor,
        data_a: torch.Tensor,
        data_b: torch.Tensor,
        temporal_order: bool = True,
    ) -> None:
        """Store a bidirectional association between two addressable patterns."""
        forward_scale = 2.0 if temporal_order else 1.0
        self._write_impl(address_a, data_b * forward_scale, apply_decay=False)
        self._write_impl(address_b, data_a, apply_decay=False)
        self.data_matrix.mul_(self.decay_rate)

    def retrieve_associates(self, cue_address: torch.Tensor) -> torch.Tensor:
        """Retrieve the strongest associated data from a partial or noisy cue."""
        return self.read(cue_address)

    def inhibit(
        self,
        activations: torch.Tensor,
        addresses: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """Keep the top-k strongest non-overlapping activations."""
        activations, squeeze = self._ensure_2d(activations.to(torch.float32))
        if addresses.dim() == 2:
            addresses = addresses.unsqueeze(0)
        if activations.shape[:1] != addresses.shape[:1] or activations.shape[1] != addresses.shape[1]:
            raise ValueError("Activation and address shapes must align on batch and pattern axes.")

        binary_addresses = self.compute_address(addresses)
        inhibited = torch.zeros_like(activations)

        for batch_idx in range(activations.shape[0]):
            order = torch.argsort(activations[batch_idx], descending=True)
            selected: List[int] = []

            for idx in order.tolist():
                if activations[batch_idx, idx] <= 0:
                    continue
                if len(selected) >= k:
                    break
                if not selected:
                    selected.append(idx)
                    inhibited[batch_idx, idx] = activations[batch_idx, idx]
                    continue

                candidate = binary_addresses[batch_idx, idx].unsqueeze(0)
                winners = binary_addresses[batch_idx, selected]
                distances = hamming_distance(candidate, winners)

                if torch.all(distances > self.hamming_radius):
                    selected.append(idx)
                    inhibited[batch_idx, idx] = activations[batch_idx, idx]

        return inhibited.squeeze(0) if squeeze else inhibited

    def get_stats(self) -> dict[str, float | int]:
        """Return basic occupancy and usage statistics."""
        occupied = self.data_matrix.abs().sum(dim=-1) > 0
        num_stored = int(occupied.sum().item())
        capacity_used = num_stored / float(self.num_hard_locations)
        return {
            "num_stored": num_stored,
            "mean_activation_count": float(self.activation_counts.mean().item()),
            "sparsity": float(1.0 - capacity_used),
            "capacity_used": float(capacity_used),
        }


class TemporalAssociator(nn.Module):
    """Forms temporally asymmetric SDM associations from recent activations."""

    def __init__(self, sdm: SparseDistributedMemory, config: SDMConfig):
        super().__init__()
        self.sdm = sdm
        self.config = config
        self.recent_activations: List[Tuple[torch.Tensor, torch.Tensor, float]] = []

    def record_activation(
        self,
        address: torch.Tensor,
        data: torch.Tensor,
        timestamp: float,
    ) -> None:
        """Add an activation to the recent temporal buffer."""
        address = self.sdm.compute_address(address)
        address, _ = self.sdm._ensure_2d(address)
        data, _ = self.sdm._ensure_2d(data.to(torch.float32))

        if address.shape[0] != data.shape[0]:
            raise ValueError(
                "Address and data batch sizes must match: "
                f"{address.shape[0]} != {data.shape[0]}"
            )

        for addr_row, data_row in zip(address, data):
            self.recent_activations.append(
                (addr_row.detach().clone(), data_row.detach().clone(), float(timestamp))
            )

        cutoff = float(timestamp) - float(self.config.stdp_window)
        self.recent_activations = [
            activation
            for activation in self.recent_activations
            if activation[2] >= cutoff
        ]

    def form_associations(self) -> None:
        """Form STDP-like associations for recent activations within the time window."""
        if len(self.recent_activations) < 2:
            return

        ordered = sorted(self.recent_activations, key=lambda item: item[2])
        for idx, (address_a, data_a, time_a) in enumerate(ordered[:-1]):
            for address_b, data_b, time_b in ordered[idx + 1:]:
                delta_t = time_b - time_a
                if delta_t <= 0 or delta_t > self.config.stdp_window:
                    continue

                strength = math.exp(-delta_t / float(self.config.stdp_window))
                self.sdm.associate(
                    address_a,
                    address_b,
                    data_a * strength,
                    data_b * strength,
                    temporal_order=True,
                )

    def clear_buffer(self) -> None:
        """Reset the temporal activation buffer."""
        self.recent_activations.clear()

