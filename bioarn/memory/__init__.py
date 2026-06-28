"""Memory systems for sparse associative recall and temporal linking."""

from .associative_fabric import (
    AssociationResult,
    AssociativeFabric,
    FabricConnectedPool,
    FabricPoolOutput,
    VoteResult,
)
from .config import SequenceMemoryConfig
from .sdm import SparseDistributedMemory, TemporalAssociator
from .sequence_memory import (
    ChunkLibrary,
    PredictiveRetrieval,
    ReplayBuffer,
    SequenceEnsembleResult,
    SequenceMemory,
    TransitionMatrix,
)

__all__ = [
    "AssociationResult",
    "AssociativeMemoryEngine",
    "AssociativeFabric",
    "ChunkLibrary",
    "FabricConnectedPool",
    "FabricPoolOutput",
    "MemoryResult",
    "PredictiveRetrieval",
    "ReplayBuffer",
    "SequenceEnsembleResult",
    "SequenceMemory",
    "SequenceMemoryConfig",
    "SparseDistributedMemory",
    "TemporalAssociator",
    "TransitionMatrix",
    "VoteResult",
]


def __getattr__(name: str):
    if name in {"AssociativeMemoryEngine", "MemoryResult"}:
        from .associative_engine import AssociativeMemoryEngine, MemoryResult

        exports = {
            "AssociativeMemoryEngine": AssociativeMemoryEngine,
            "MemoryResult": MemoryResult,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
