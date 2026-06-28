"""Predictive coding engine: hierarchical prediction error minimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
    from bioarn.predictive.lateral_prediction import LateralPredictionNetwork

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
    "LateralPredictionNetwork",
    "PredictionQualityOutput",
    "PredictionErrorGate",
    "PoolEntropyEstimator",
    "PrecisionSignal",
    "PrecisionWeightedGate",
    "PredictiveHierarchy",
    "ResonanceLoopOutput",
    "free_energy",
]


def __getattr__(name: str):
    if name in {"ErrorGatingOutput", "PredictionErrorGate"}:
        from bioarn.predictive.error_gating import ErrorGatingOutput, PredictionErrorGate

        exports = {
            "ErrorGatingOutput": ErrorGatingOutput,
            "PredictionErrorGate": PredictionErrorGate,
        }
        return exports[name]
    if name in {
        "ActionSignal",
        "HierarchyConnector",
        "HierarchyGenerationOutput",
        "HierarchyPerceptionOutput",
        "PredictionQualityOutput",
        "PredictiveHierarchy",
        "ResonanceLoopOutput",
    }:
        from bioarn.predictive.hierarchy import (
            ActionSignal,
            HierarchyConnector,
            HierarchyGenerationOutput,
            HierarchyPerceptionOutput,
            PredictionQualityOutput,
            PredictiveHierarchy,
            ResonanceLoopOutput,
        )

        exports = {
            "ActionSignal": ActionSignal,
            "HierarchyConnector": HierarchyConnector,
            "HierarchyGenerationOutput": HierarchyGenerationOutput,
            "HierarchyPerceptionOutput": HierarchyPerceptionOutput,
            "PredictionQualityOutput": PredictionQualityOutput,
            "PredictiveHierarchy": PredictiveHierarchy,
            "ResonanceLoopOutput": ResonanceLoopOutput,
        }
        return exports[name]
    if name in {"PCStack", "PCStackOutput", "PCLayer", "PCLayerOutput", "free_energy"}:
        from bioarn.predictive.pc_layer import PCStack, PCStackOutput, PCLayer, PCLayerOutput, free_energy

        exports = {
            "PCStack": PCStack,
            "PCStackOutput": PCStackOutput,
            "PCLayer": PCLayer,
            "PCLayerOutput": PCLayerOutput,
            "free_energy": free_energy,
        }
        return exports[name]
    if name == "LateralPredictionNetwork":
        from bioarn.predictive.lateral_prediction import LateralPredictionNetwork

        return LateralPredictionNetwork
    if name in {"PoolEntropyEstimator", "PrecisionSignal", "PrecisionWeightedGate"}:
        from bioarn.predictive.precision_weighting import (
            PoolEntropyEstimator,
            PrecisionSignal,
            PrecisionWeightedGate,
        )

        exports = {
            "PoolEntropyEstimator": PoolEntropyEstimator,
            "PrecisionSignal": PrecisionSignal,
            "PrecisionWeightedGate": PrecisionWeightedGate,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
