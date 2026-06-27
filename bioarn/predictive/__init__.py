"""Predictive coding engine: hierarchical prediction error minimization."""

from bioarn.predictive.error_gating import ErrorGatingOutput, PredictionErrorGate
from bioarn.predictive.hierarchy import (
    ActionSignal,
    HierarchyConnector,
    HierarchyGenerationOutput,
    HierarchyPerceptionOutput,
    PredictionQualityOutput,
    PredictiveHierarchy,
    ResonanceLoopOutput,
)
from bioarn.predictive.pc_layer import PCStack, PCStackOutput, PCLayer, PCLayerOutput, free_energy
from bioarn.predictive.precision_weighting import (
    PoolEntropyEstimator,
    PrecisionSignal,
    PrecisionWeightedGate,
)

__all__ = [
    "ActionSignal",
    "ErrorGatingOutput",
    "HierarchyConnector",
    "HierarchyGenerationOutput",
    "HierarchyPerceptionOutput",
    "PCLayer",
    "PCLayerOutput",
    "PCStack",
    "PCStackOutput",
    "PredictionQualityOutput",
    "PredictionErrorGate",
    "PoolEntropyEstimator",
    "PrecisionSignal",
    "PrecisionWeightedGate",
    "PredictiveHierarchy",
    "ResonanceLoopOutput",
    "free_energy",
]
