"""Bio-ARN public package interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bioarn.__version__ import __version__

if TYPE_CHECKING:
    from bioarn.config import CCCConfig, BioARNConfig, ConvCCCConfig, PrecisionConfig
    from bioarn.core.ccc import CCCPool, ConceptCellCluster
    from bioarn.core.conv_ccc import ConvCCCPool
    from bioarn.loop import SensorimotorLoop
    from bioarn.system import BioARNCore
    from bioarn.training import VisionTrainConfig, VisionTrainer

__all__ = [
    "BioARNConfig",
    "BioARNCore",
    "CCCConfig",
    "CCCPool",
    "ConceptCellCluster",
    "ConvCCCConfig",
    "ConvCCCPool",
    "PrecisionConfig",
    "SensorimotorLoop",
    "VisionTrainConfig",
    "VisionTrainer",
    "__version__",
]


def __getattr__(name: str):
    if name in {"CCCConfig", "BioARNConfig", "ConvCCCConfig", "PrecisionConfig"}:
        from bioarn.config import CCCConfig, BioARNConfig, ConvCCCConfig, PrecisionConfig

        exports = {
            "CCCConfig": CCCConfig,
            "BioARNConfig": BioARNConfig,
            "ConvCCCConfig": ConvCCCConfig,
            "PrecisionConfig": PrecisionConfig,
        }
        return exports[name]
    if name in {"CCCPool", "ConceptCellCluster"}:
        from bioarn.core.ccc import CCCPool, ConceptCellCluster

        exports = {
            "CCCPool": CCCPool,
            "ConceptCellCluster": ConceptCellCluster,
        }
        return exports[name]
    if name == "ConvCCCPool":
        from bioarn.core.conv_ccc import ConvCCCPool

        return ConvCCCPool
    if name == "SensorimotorLoop":
        from bioarn.loop import SensorimotorLoop

        return SensorimotorLoop
    if name == "BioARNCore":
        from bioarn.system import BioARNCore

        return BioARNCore
    if name in {"VisionTrainConfig", "VisionTrainer"}:
        from bioarn.training import VisionTrainConfig, VisionTrainer

        exports = {
            "VisionTrainConfig": VisionTrainConfig,
            "VisionTrainer": VisionTrainer,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
