"""Training exports for Bio-ARN."""

from bioarn.training.trainer import EvalResult, OnlineTrainer, TrainResult
from bioarn.training.text_training import (
    GenerationMetrics,
    TextGenConfig,
    TextGenerationTrainer,
    TrainingMetrics,
    build_builtin_corpus,
)
from bioarn.training.vision_training import (
    SyntheticCIFAR10Stream,
    VisionTrainConfig,
    VisionTrainer,
    load_cifar10_or_synthetic,
    take_samples,
)

Trainer = OnlineTrainer

__all__ = [
    "EvalResult",
    "GenerationMetrics",
    "OnlineTrainer",
    "Trainer",
    "SyntheticCIFAR10Stream",
    "TextGenConfig",
    "TextGenerationTrainer",
    "TrainResult",
    "TrainingMetrics",
    "VisionTrainConfig",
    "VisionTrainer",
    "build_builtin_corpus",
    "load_cifar10_or_synthetic",
    "take_samples",
]
