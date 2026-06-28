"""Lazy multimodal exports for Bio-ARN."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bioarn.config import MultimodalFusionConfig
    from bioarn.multimodal.alignment import AlignmentMetrics, ModalityAligner
    from bioarn.multimodal.captioning import SpikeCaptioner
    from bioarn.multimodal.config import MultimodalConfig
    from bioarn.multimodal.fusion import (
        CrossModalAssociation,
        ModalityResult,
        MultimodalFusion,
        MultimodalFusionEngine,
        MultimodalInput,
        MultimodalOutput,
    )

__all__ = [
    "AlignmentMetrics",
    "CrossModalAssociation",
    "ModalityAligner",
    "ModalityResult",
    "MultimodalConfig",
    "MultimodalFusion",
    "MultimodalFusionConfig",
    "MultimodalFusionEngine",
    "MultimodalInput",
    "MultimodalOutput",
    "SpikeCaptioner",
]


def __getattr__(name: str):
    if name in {"MultimodalConfig"}:
        from bioarn.multimodal.config import MultimodalConfig

        return MultimodalConfig
    if name in {"MultimodalFusionConfig"}:
        from bioarn.config import MultimodalFusionConfig

        return MultimodalFusionConfig
    if name in {"AlignmentMetrics", "ModalityAligner"}:
        from bioarn.multimodal.alignment import AlignmentMetrics, ModalityAligner

        return {
            "AlignmentMetrics": AlignmentMetrics,
            "ModalityAligner": ModalityAligner,
        }[name]
    if name in {"SpikeCaptioner"}:
        from bioarn.multimodal.captioning import SpikeCaptioner

        return SpikeCaptioner
    if name in {
        "CrossModalAssociation",
        "ModalityResult",
        "MultimodalFusion",
        "MultimodalFusionEngine",
        "MultimodalInput",
        "MultimodalOutput",
    }:
        from bioarn.multimodal.fusion import (
            CrossModalAssociation,
            ModalityResult,
            MultimodalFusion,
            MultimodalFusionEngine,
            MultimodalInput,
            MultimodalOutput,
        )

        return {
            "CrossModalAssociation": CrossModalAssociation,
            "ModalityResult": ModalityResult,
            "MultimodalFusion": MultimodalFusion,
            "MultimodalFusionEngine": MultimodalFusionEngine,
            "MultimodalInput": MultimodalInput,
            "MultimodalOutput": MultimodalOutput,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
