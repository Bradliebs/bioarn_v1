"""Core components: spiking neurons, margin gates, concept cell clusters."""

from bioarn.core.ccc import CCCOutput, CCCPool, CCCPoolOutput, ConceptCellCluster
from bioarn.core.consolidation import SynapticConsolidation
from bioarn.core.margin_gate import MarginGate, MarginGateOutput, ResonanceOutput
from bioarn.core.spiking import (
    LIFLayer,
    LIFNeuron,
    SurrogateSpike,
    delta_encode,
    firing_rate,
    interspike_interval,
    latency_encode,
    rate_encode,
    spike_count,
)
from bioarn.core.stdp import STDPRule

__all__ = [
    "CCCOutput",
    "CCCPool",
    "CCCPoolOutput",
    "ConceptCellCluster",
    "SynapticConsolidation",
    "LIFLayer",
    "LIFNeuron",
    "MarginGate",
    "MarginGateOutput",
    "ResonanceOutput",
    "SurrogateSpike",
    "STDPRule",
    "delta_encode",
    "firing_rate",
    "interspike_interval",
    "latency_encode",
    "rate_encode",
    "spike_count",
]
