"""Training exports for Bio-ARN."""

from typing import TYPE_CHECKING

from bioarn.config import AudioTrainConfig
from bioarn.training.audio_training import AudioTrainer
from bioarn.training.ensemble_training import (
    AugmentFn,
    EnsembleTrainMetrics,
    EnsembleTrainer,
    ExpertTrainMetrics,
)
from bioarn.training.trainer import EvalResult, OnlineTrainer, TrainResult
from bioarn.training.text_training import (
    GenerationMetrics,
    TextGenConfig,
    TextGenerationTrainer,
    TrainingMetrics,
    build_builtin_corpus,
)
from bioarn.training.curriculum import CurriculumScheduler
from bioarn.training.maturation import MaturationConfig, MaturationSchedule
from bioarn.training.rl_training import RLTrainer, TrainingResult as RLTrainingResult
from bioarn.training.temporal_training import (
    FrameTemporalResult,
    SequenceResult,
    TemporalTrainer,
)
from bioarn.training.vision_training import (
    SyntheticCIFAR10Stream,
    VisionTrainConfig,
    VisionTrainer,
    load_cifar10_or_synthetic,
    take_samples,
)

if TYPE_CHECKING:
    from bioarn.training.multimodal_training import (
        MultimodalExample,
        MultimodalTrainer,
        MultimodalTrainingResult,
    )

Trainer = OnlineTrainer

__all__ = [
    "AudioTrainConfig",
    "AudioTrainer",
    "AugmentFn",
    "EnsembleTrainMetrics",
    "EnsembleTrainer",
    "EvalResult",
    "ExpertTrainMetrics",
    "GenerationMetrics",
    "MaturationConfig",
    "MaturationSchedule",
    "MultimodalExample",
    "MultimodalTrainer",
    "MultimodalTrainingResult",
    "FrameTemporalResult",
    "RLTrainer",
    "RLTrainingResult",
    "OnlineTrainer",
    "SequenceResult",
    "TemporalTrainer",
    "Trainer",
    "SyntheticCIFAR10Stream",
    "TextGenConfig",
    "TextGenerationTrainer",
    "TrainResult",
    "TrainingMetrics",
    "VisionTrainConfig",
    "VisionTrainer",
    "build_builtin_corpus",
    "CurriculumScheduler",
    "load_cifar10_or_synthetic",
    "take_samples",
]


def __getattr__(name: str):
    if name in {"MultimodalExample", "MultimodalTrainer", "MultimodalTrainingResult"}:
        from bioarn.training.multimodal_training import (
            MultimodalExample,
            MultimodalTrainer,
            MultimodalTrainingResult,
        )

        exports = {
            "MultimodalExample": MultimodalExample,
            "MultimodalTrainer": MultimodalTrainer,
            "MultimodalTrainingResult": MultimodalTrainingResult,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
