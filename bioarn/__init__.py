"""Bio-ARN public package interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bioarn.__version__ import __version__

if TYPE_CHECKING:
    from bioarn.config import (
        AugmentationConfig,
        AudioConfig,
        AudioHierarchyConfig,
        AudioTrainConfig,
        AgentConfig,
        AssociativeMemoryConfig,
        BioARNConfig,
        CCCConfig,
        ConvCCCConfig,
        PrecisionConfig,
        RLTrainConfig,
        WhiteningConfig,
        WorldModelConfig,
    )
    from bioarn.core.ccc import CCCPool, ConceptCellCluster
    from bioarn.core.conv_ccc import ConvCCCPool
    from bioarn.loop import SensorimotorLoop
    from bioarn.memory.associative_engine import AssociativeMemoryEngine, MemoryResult
    from bioarn.rl import BioARNAgent, BioARNWorldModel
    from bioarn.system import BioARNCore
    from bioarn.training import AudioTrainer, RLTrainer, VisionTrainConfig, VisionTrainer

__all__ = [
    "AudioConfig",
    "AudioHierarchyConfig",
    "AudioTrainConfig",
    "AudioTrainer",
    "AugmentationConfig",
    "AgentConfig",
    "AssociativeMemoryConfig",
    "AssociativeMemoryEngine",
    "BioARNAgent",
    "BioARNWorldModel",
    "BioARNConfig",
    "BioARNCore",
    "CCCConfig",
    "CCCPool",
    "ConceptCellCluster",
    "ConvCCCConfig",
    "ConvCCCPool",
    "MemoryResult",
    "PrecisionConfig",
    "RLTrainConfig",
    "RLTrainer",
    "SensorimotorLoop",
    "VisionTrainConfig",
    "VisionTrainer",
    "WhiteningConfig",
    "WorldModelConfig",
    "__version__",
]


def __getattr__(name: str):
    if name in {
        "AugmentationConfig",
        "AudioConfig",
        "AudioHierarchyConfig",
        "AudioTrainConfig",
        "AgentConfig",
        "AssociativeMemoryConfig",
        "CCCConfig",
        "BioARNConfig",
        "ConvCCCConfig",
        "PrecisionConfig",
        "RLTrainConfig",
        "WhiteningConfig",
        "WorldModelConfig",
    }:
        from bioarn.config import (
            AugmentationConfig,
            AudioConfig,
            AudioHierarchyConfig,
            AudioTrainConfig,
            AgentConfig,
            AssociativeMemoryConfig,
            BioARNConfig,
            CCCConfig,
            ConvCCCConfig,
            PrecisionConfig,
            RLTrainConfig,
            WhiteningConfig,
            WorldModelConfig,
        )

        exports = {
            "AugmentationConfig": AugmentationConfig,
            "AudioConfig": AudioConfig,
            "AudioHierarchyConfig": AudioHierarchyConfig,
            "AudioTrainConfig": AudioTrainConfig,
            "AgentConfig": AgentConfig,
            "AssociativeMemoryConfig": AssociativeMemoryConfig,
            "CCCConfig": CCCConfig,
            "BioARNConfig": BioARNConfig,
            "ConvCCCConfig": ConvCCCConfig,
            "PrecisionConfig": PrecisionConfig,
            "RLTrainConfig": RLTrainConfig,
            "WhiteningConfig": WhiteningConfig,
            "WorldModelConfig": WorldModelConfig,
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
    if name in {"AssociativeMemoryEngine", "MemoryResult"}:
        from bioarn.memory.associative_engine import AssociativeMemoryEngine, MemoryResult

        exports = {
            "AssociativeMemoryEngine": AssociativeMemoryEngine,
            "MemoryResult": MemoryResult,
        }
        return exports[name]
    if name in {"BioARNAgent", "BioARNWorldModel"}:
        from bioarn.rl import BioARNAgent, BioARNWorldModel

        exports = {
            "BioARNAgent": BioARNAgent,
            "BioARNWorldModel": BioARNWorldModel,
        }
        return exports[name]
    if name == "BioARNCore":
        from bioarn.system import BioARNCore

        return BioARNCore
    if name in {"AudioTrainer", "RLTrainer", "VisionTrainConfig", "VisionTrainer"}:
        from bioarn.training import AudioTrainer, RLTrainer, VisionTrainConfig, VisionTrainer

        exports = {
            "AudioTrainer": AudioTrainer,
            "RLTrainer": RLTrainer,
            "VisionTrainConfig": VisionTrainConfig,
            "VisionTrainer": VisionTrainer,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
