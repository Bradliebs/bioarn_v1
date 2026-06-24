"""Mock and optional real Lava processes for Bio-ARN deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from bioarn.config import MarginGateConfig, SpikingConfig
from bioarn.core.ccc import ConceptCellCluster
from bioarn.core.margin_gate import MarginGateOutput
from bioarn.core.math_utils import cosine_similarity, normalize, sparse_top_k
from bioarn.memory.sdm import SparseDistributedMemory

try:
    import lava.lib.dl as lava_dl  # type: ignore[import-not-found]
    from lava.magma.core.run_conditions import RunSteps  # type: ignore[import-not-found]
    from lava.magma.core.run_configs import Loihi2SimCfg  # type: ignore[import-not-found]
    from lava.proc.dense.process import Dense  # type: ignore[import-not-found]
    from lava.proc.lif.process import LIF  # type: ignore[import-not-found]

    LAVA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    lava_dl = None
    RunSteps = None
    Loihi2SimCfg = None
    Dense = None
    LIF = None
    LAVA_AVAILABLE = False


class MockLavaProcess:
    """Stand-in for a Lava process when the SDK is unavailable."""

    def __init__(self, name: str, process_type: str = "mock") -> None:
        self.name = name
        self.process_type = process_type

    def reset_state(self) -> None:
        """Reset any internal process state."""

    def describe(self) -> dict[str, Any]:
        """Return a serializable process description."""

        return {
            "name": self.name,
            "type": self.process_type,
            "lava_available": LAVA_AVAILABLE,
        }


class DenseLavaProcess(MockLavaProcess):
    """Dense matrix projection that mirrors Lava Dense behavior."""

    def __init__(
        self,
        weights: torch.Tensor,
        bias: torch.Tensor | None = None,
        *,
        name: str,
    ) -> None:
        super().__init__(name=name, process_type="dense")
        self.weights = weights.detach().clone().to(torch.float32)
        self.bias = bias.detach().clone().to(torch.float32) if bias is not None else None
        self.real_process = None
        if LAVA_AVAILABLE and Dense is not None:  # pragma: no cover - exercised only with Lava installed
            try:
                self.real_process = Dense(weights=self.weights.detach().cpu().numpy())
            except Exception:
                self.real_process = None

    def step(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a dense projection."""

        return F.linear(inputs.to(self.weights.dtype), self.weights, self.bias)

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "weight_shape": list(self.weights.shape),
                "bias": self.bias is not None,
            }
        )
        return payload


class MockLIFProcess(MockLavaProcess):
    """Pure-Python LIF process with Bio-ARN-compatible dynamics."""

    def __init__(
        self,
        num_neurons: int,
        config: SpikingConfig | None = None,
        *,
        name: str,
    ) -> None:
        super().__init__(name=name, process_type="lif")
        config = config or SpikingConfig()
        self.num_neurons = int(num_neurons)
        self.beta = float(config.beta)
        self.threshold = float(config.threshold)
        self.reset = float(config.reset)
        self.dt = float(config.dt)
        self.refractory_steps = int(config.refractory_steps)
        self.voltage = torch.empty(0, dtype=torch.float32)
        self.refractory_counter = torch.empty(0, dtype=torch.long)
        self.real_process = None
        if LAVA_AVAILABLE and LIF is not None:  # pragma: no cover - exercised only with Lava installed
            try:
                self.real_process = LIF(
                    shape=(self.num_neurons,),
                    du=int(round((1.0 - self.beta) * 4096)),
                    dv=int(round((1.0 - self.beta) * 4096)),
                    vth=int(round(self.threshold * 256)),
                )
            except Exception:
                self.real_process = None

    def _ensure_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        shape = (batch_size, self.num_neurons)
        if self.voltage.shape != shape or self.voltage.device != device or self.voltage.dtype != dtype:
            self.voltage = torch.full(shape, self.reset, device=device, dtype=dtype)
        if self.refractory_counter.shape != shape or self.refractory_counter.device != device:
            self.refractory_counter = torch.zeros(shape, device=device, dtype=torch.long)

    def reset_state(
        self,
        batch_size: int | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Reset the membrane and refractory state."""

        if batch_size is None:
            device = device or torch.device("cpu")
            dtype = dtype or torch.float32
            self.voltage = torch.empty(0, device=device, dtype=dtype)
            self.refractory_counter = torch.empty(0, device=device, dtype=torch.long)
            return
        self._ensure_state(
            batch_size=batch_size,
            device=device or torch.device("cpu"),
            dtype=dtype or torch.float32,
        )
        self.voltage.fill_(self.reset)
        self.refractory_counter.zero_()

    def _forward_step(self, input_current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input_current.dim() != 2:
            raise ValueError("Single-step current must have shape (batch, neurons).")
        if input_current.shape[1] != self.num_neurons:
            raise ValueError(
                f"Expected {self.num_neurons} neurons, received {input_current.shape[1]}."
            )

        self._ensure_state(
            batch_size=input_current.shape[0],
            device=input_current.device,
            dtype=input_current.dtype,
        )
        refractory_mask = self.refractory_counter > 0
        integrated = self.beta * self.voltage + (self.dt * input_current)
        reset_value = torch.full_like(integrated, self.reset)
        integrated = torch.where(refractory_mask, reset_value, integrated)
        spike = ((integrated - self.threshold) > 0).to(input_current.dtype)
        spike = spike * (~refractory_mask).to(input_current.dtype)
        next_voltage = torch.where(spike > 0, reset_value, integrated)

        next_refractory = torch.clamp(self.refractory_counter - 1, min=0)
        if self.refractory_steps > 0:
            next_refractory = torch.where(
                spike > 0,
                torch.full_like(next_refractory, self.refractory_steps),
                next_refractory,
            )
        self.voltage = next_voltage.detach().clone()
        self.refractory_counter = next_refractory.detach().clone()
        return spike, next_voltage

    def forward(self, input_current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one step or a full time-series through the LIF process."""

        if input_current.dim() == 2:
            return self._forward_step(input_current)
        if input_current.dim() != 3:
            raise ValueError(
                "Input current must have shape (batch, neurons) or (time, batch, neurons)."
            )
        spikes = []
        voltages = []
        for current_t in input_current.unbind(dim=0):
            spike_t, voltage_t = self._forward_step(current_t)
            spikes.append(spike_t)
            voltages.append(voltage_t)
        return torch.stack(spikes, dim=0), torch.stack(voltages, dim=0)

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "num_neurons": self.num_neurons,
                "beta": self.beta,
                "threshold": self.threshold,
                "reset": self.reset,
                "refractory_steps": self.refractory_steps,
            }
        )
        return payload


class MarginGateLavaProcess(MockLavaProcess):
    """Lava-style process implementing Bio-ARN's cosine margin gate."""

    def __init__(
        self,
        config: MarginGateConfig,
        concept_direction: torch.Tensor | None = None,
        *,
        name: str = "margin_gate",
    ) -> None:
        super().__init__(name=name, process_type="margin_gate")
        self.theta_margin = float(config.theta_margin)
        self.theta_resonance = float(config.theta_resonance)
        self.concept_direction = (
            normalize(concept_direction.reshape(1, -1)).squeeze(0).to(torch.float32)
            if concept_direction is not None and concept_direction.numel() > 0
            else None
        )

    def step(
        self,
        input_activation: torch.Tensor,
        concept_direction: torch.Tensor | None = None,
    ) -> MarginGateOutput:
        """Gate a concept activation using cosine similarity."""

        if input_activation.dim() == 1:
            input_activation = input_activation.unsqueeze(0)
        direction = concept_direction
        if direction is None:
            direction = self.concept_direction
        if direction is None:
            raise ValueError("MarginGateLavaProcess requires a concept direction.")
        if direction.dim() == 1:
            direction = direction.unsqueeze(0).expand(input_activation.shape[0], -1)
        confidence = cosine_similarity(input_activation, direction.to(input_activation.dtype))
        fired = confidence > self.theta_margin
        output = torch.where(
            fired.unsqueeze(-1),
            input_activation,
            torch.zeros_like(input_activation),
        )
        return MarginGateOutput(
            output=output,
            confidence=confidence,
            fired=fired,
            abstained=~fired,
        )

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "theta_margin": self.theta_margin,
                "theta_resonance": self.theta_resonance,
                "concept_dim": 0 if self.concept_direction is None else int(self.concept_direction.numel()),
            }
        )
        return payload


class CCCLavaProcess(MockLavaProcess):
    """Lava-ready wrapper for one Bio-ARN concept cell cluster."""

    def __init__(
        self,
        ccc: ConceptCellCluster,
        spiking_config: SpikingConfig | None = None,
        *,
        name: str,
    ) -> None:
        super().__init__(name=name, process_type="ccc")
        self.ccc = ccc
        self.config = ccc.config
        self.f1_dense = DenseLavaProcess(
            ccc.f1_layer.weight.detach(),
            ccc.f1_layer.bias.detach() if ccc.f1_layer.bias is not None else None,
            name=f"{name}.f1_dense",
        )
        self.f2_dense = DenseLavaProcess(
            ccc.f2_weights.detach(),
            name=f"{name}.f2_dense",
        )
        self.f2_lif = MockLIFProcess(
            num_neurons=self.config.concept_dim,
            config=spiking_config,
            name=f"{name}.f2_lif",
        )
        self.margin_gate = MarginGateLavaProcess(
            MarginGateConfig(
                theta_margin=float(ccc.margin_gate.theta_margin.item()),
                theta_margin_lr=float(ccc.margin_gate.theta_margin_lr),
                theta_resonance=float(ccc.margin_gate.theta_resonance.item()),
            ),
            concept_direction=ccc.concept_direction.detach(),
            name=f"{name}.margin_gate",
        )
        self.feedback_dense = DenseLavaProcess(
            ccc.feedback_weights.detach(),
            name=f"{name}.feedback_dense",
        )

    @staticmethod
    def _ensure_batch(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if tensor.dim() == 1:
            return tensor.unsqueeze(0), True
        if tensor.dim() != 2:
            raise ValueError("CCC inputs must have shape (input_dim,) or (batch, input_dim).")
        return tensor, False

    def subprocesses(self) -> list[MockLavaProcess]:
        """Return the concrete processes used inside this CCC wrapper."""

        return [self.f1_dense, self.f2_dense, self.f2_lif, self.margin_gate, self.feedback_dense]

    def connections(self) -> list[dict[str, str]]:
        """Return a serializable connection list."""

        return [
            {"from": self.f1_dense.name, "to": self.f2_dense.name, "signal": "f1_features"},
            {"from": self.f2_dense.name, "to": self.f2_lif.name, "signal": "concept_current"},
            {"from": self.f2_lif.name, "to": self.margin_gate.name, "signal": "concept_activation"},
            {"from": self.margin_gate.name, "to": self.feedback_dense.name, "signal": "gated_output"},
            {"from": self.feedback_dense.name, "to": self.margin_gate.name, "signal": "prediction_feedback"},
        ]

    def run(self, raw_input: torch.Tensor, num_timesteps: int = 4) -> dict[str, Any]:
        """Run the CCC on the mock Lava stack."""

        raw_batch, squeeze = self._ensure_batch(raw_input.to(torch.float32))
        f1_linear = self.f1_dense.step(raw_batch)
        f1_output = sparse_top_k(F.relu(f1_linear), self.config.f1_top_k)
        f2_current = self.f2_dense.step(f1_output)

        self.f2_lif.reset_state(
            batch_size=raw_batch.shape[0],
            device=raw_batch.device,
            dtype=raw_batch.dtype,
        )
        repeated_current = f2_current.unsqueeze(0).expand(max(int(num_timesteps), 1), -1, -1).clone()
        f2_spikes, f2_voltage = self.f2_lif.forward(repeated_current)

        if not bool(self.ccc.is_committed.item()):
            confidence = torch.zeros(raw_batch.shape[0], dtype=raw_batch.dtype, device=raw_batch.device)
            gate_output = MarginGateOutput(
                output=torch.zeros_like(f2_current),
                confidence=confidence,
                fired=torch.zeros_like(confidence, dtype=torch.bool),
                abstained=torch.ones_like(confidence, dtype=torch.bool),
            )
            prediction = torch.zeros(
                raw_batch.shape[0],
                self.config.num_f1_features,
                device=raw_batch.device,
                dtype=raw_batch.dtype,
            )
        else:
            gate_output = self.margin_gate.step(f2_current, self.ccc.concept_direction.detach())
            prediction = self.feedback_dense.step(gate_output.output)

        output = {
            "fired": bool(gate_output.fired.any().item()),
            "abstained": bool(gate_output.abstained.all().item()),
            "confidence": gate_output.confidence.squeeze(0) if squeeze else gate_output.confidence,
            "f1_output": f1_output.squeeze(0) if squeeze else f1_output,
            "f2_activation": f2_current.squeeze(0) if squeeze else f2_current,
            "gate_output": gate_output.output.squeeze(0) if squeeze else gate_output.output,
            "prediction": prediction.squeeze(0) if squeeze else prediction,
            "spikes": f2_spikes,
            "voltages": f2_voltage,
            "spike_counts": f2_spikes.sum(dim=0).squeeze(0) if squeeze else f2_spikes.sum(dim=0),
        }
        output["output"] = output["gate_output"]
        return output

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "input_dim": int(self.config.input_dim),
                "num_f1_features": int(self.config.num_f1_features),
                "concept_dim": int(self.config.concept_dim),
                "committed": bool(self.ccc.is_committed.item()),
                "subprocesses": [process.describe() for process in self.subprocesses()],
            }
        )
        return payload


class SDMLavaProcess(MockLavaProcess):
    """Lava-style sparse distributed memory process."""

    def __init__(self, sdm: SparseDistributedMemory, *, name: str = "sdm") -> None:
        super().__init__(name=name, process_type="sdm")
        self.sdm = sdm
        self.address_dense = DenseLavaProcess(
            sdm.hard_locations.detach(),
            name=f"{name}.address_dense",
        )
        self.content_dense = DenseLavaProcess(
            sdm.data_matrix.detach().transpose(0, 1),
            name=f"{name}.content_dense",
        )

    @staticmethod
    def _ensure_batch(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if tensor.dim() == 1:
            return tensor.unsqueeze(0), True
        return tensor, False

    def _refresh_content_dense(self) -> None:
        self.content_dense.weights = self.sdm.data_matrix.detach().transpose(0, 1).to(torch.float32)

    def write(self, address: torch.Tensor, data: torch.Tensor) -> None:
        """Store content in the backing SDM."""

        self.sdm.write(address, data)
        self._refresh_content_dense()

    def read(self, cue: torch.Tensor, num_timesteps: int = 3) -> dict[str, Any]:
        """Retrieve content using an SDM cue."""

        del num_timesteps
        binary_address = self.sdm.compute_address(cue)
        address_batch, squeeze = self._ensure_batch(binary_address.to(torch.float32))
        overlaps = self.address_dense.step(address_batch)
        address_ones = address_batch.sum(dim=-1, keepdim=True)
        location_ones = self.sdm.hard_locations.to(address_batch.dtype).sum(dim=-1).unsqueeze(0)
        distances = address_ones + location_ones - (2.0 * overlaps)
        activated = distances <= float(self.sdm.hamming_radius)
        self._refresh_content_dense()
        raw_retrieved = self.content_dense.step(activated.to(torch.float32))
        counts = activated.to(torch.float32) @ self.sdm.activation_counts.to(torch.float32)
        retrieved = raw_retrieved / counts.clamp_min(1.0).unsqueeze(-1)
        output = {
            "address": address_batch.squeeze(0) if squeeze else address_batch,
            "distances": distances.squeeze(0) if squeeze else distances,
            "activated": activated.squeeze(0) if squeeze else activated,
            "retrieved": retrieved.squeeze(0) if squeeze else retrieved,
            "spike_counts": activated.to(torch.float32).sum(dim=-1).squeeze(0)
            if squeeze
            else activated.to(torch.float32).sum(dim=-1),
            "output": retrieved.squeeze(0) if squeeze else retrieved,
        }
        return output

    def run(self, input_data: dict[str, torch.Tensor] | torch.Tensor, num_timesteps: int = 3) -> dict[str, Any]:
        """Optionally write, then read from the SDM."""

        if isinstance(input_data, dict):
            address = input_data.get("address")
            data = input_data.get("data")
            if input_data.get("write") and address is not None and data is not None:
                self.write(address, data)
            if address is None:
                raise ValueError("SDM Lava input must include an 'address' tensor.")
            return self.read(address, num_timesteps=num_timesteps)
        return self.read(input_data, num_timesteps=num_timesteps)

    def connections(self) -> list[dict[str, str]]:
        """Return the internal SDM topology."""

        return [
            {"from": self.address_dense.name, "to": self.content_dense.name, "signal": "activated_locations"},
        ]

    def describe(self) -> dict[str, Any]:
        payload = super().describe()
        payload.update(
            {
                "address_dim": int(self.sdm.address_dim),
                "num_hard_locations": int(self.sdm.num_hard_locations),
                "data_dim": int(self.sdm.data_dim),
                "hamming_radius": int(self.sdm.hamming_radius),
                "subprocesses": [self.address_dense.describe(), self.content_dense.describe()],
            }
        )
        return payload


__all__ = [
    "CCCLavaProcess",
    "DenseLavaProcess",
    "LAVA_AVAILABLE",
    "MarginGateLavaProcess",
    "MockLIFProcess",
    "MockLavaProcess",
    "SDMLavaProcess",
]
