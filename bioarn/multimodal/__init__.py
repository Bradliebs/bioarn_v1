"""Multimodal fusion package for Bio-ARN."""

from bioarn.multimodal.alignment import AlignmentMetrics, ModalityAligner
from bioarn.multimodal.captioning import SpikeCaptioner
from bioarn.multimodal.config import MultimodalConfig
from bioarn.multimodal.fusion import CrossModalAssociation, MultimodalFusion

__all__ = [
    "AlignmentMetrics",
    "CrossModalAssociation",
    "ModalityAligner",
    "MultimodalConfig",
    "MultimodalFusion",
    "SpikeCaptioner",
]
