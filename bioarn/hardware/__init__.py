"""Neuromorphic hardware abstraction layer for Bio-ARN."""

from bioarn.hardware.backend import (
    ComparisonReport,
    ComponentMapper,
    EnergyEstimate,
    HardwareInfo,
    LatencyEstimate,
    MappedComponent,
    NeuromorphicBackend,
    NeuronGroupHandle,
    PowerEstimate,
    StepResult,
    SynapseHandle,
    SystemMapping,
)
from bioarn.hardware.energy_model import (
    BrainComparisonReport,
    EnergyBreakdown,
    EnergyComparisonReport,
    EnergyModel,
)
from bioarn.hardware.lava_bridge import (
    DeploymentPackage,
    EquivalenceReport,
    HardwareRequirements,
    LavaBridge,
    LavaProcessGraph,
)
from bioarn.hardware.lava_processes import (
    CCCLavaProcess,
    LAVA_AVAILABLE,
    MarginGateLavaProcess,
    MockLIFProcess,
    MockLavaProcess,
    SDMLavaProcess,
)
from bioarn.hardware.loihi_backend import LoihiBackend
from bioarn.hardware.loihi_port import (
    FunctionalEquivalenceValidator,
    LoihiCCCMapping,
    LoihiGNWMapping,
    LoihiMapping,
    LoihiNeuronSpec,
    LoihiPEMapping,
    LoihiSDMMapping,
    LoihiSystemMapping,
    SystemValidationResult,
    ValidationResult,
)
from bioarn.hardware.profiler import HardwareProfiler
from bioarn.hardware.pytorch_backend import PyTorchBackend
from bioarn.hardware.spec import ASICSpec

__all__ = [
    "ASICSpec",
    "BrainComparisonReport",
    "ComparisonReport",
    "ComponentMapper",
    "EnergyBreakdown",
    "EnergyEstimate",
    "EnergyComparisonReport",
    "EnergyModel",
    "CCCLavaProcess",
    "DeploymentPackage",
    "EquivalenceReport",
    "FunctionalEquivalenceValidator",
    "HardwareInfo",
    "HardwareRequirements",
    "HardwareProfiler",
    "LAVA_AVAILABLE",
    "LatencyEstimate",
    "LavaBridge",
    "LavaProcessGraph",
    "LoihiBackend",
    "LoihiCCCMapping",
    "LoihiGNWMapping",
    "LoihiMapping",
    "LoihiNeuronSpec",
    "LoihiPEMapping",
    "LoihiSDMMapping",
    "LoihiSystemMapping",
    "MarginGateLavaProcess",
    "MappedComponent",
    "MockLIFProcess",
    "MockLavaProcess",
    "NeuromorphicBackend",
    "NeuronGroupHandle",
    "PowerEstimate",
    "PyTorchBackend",
    "SDMLavaProcess",
    "StepResult",
    "SynapseHandle",
    "SystemValidationResult",
    "SystemMapping",
    "ValidationResult",
]
