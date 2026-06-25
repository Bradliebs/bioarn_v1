"""Ensemble-aware online training for Bio-ARN with diversity tracking and Hebbian boosting."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch

from bioarn.ensemble import DiversityManager, EnsemblePool


AugmentFn = Callable[[torch.Tensor, int], torch.Tensor]
"""Signature: (flat_image, expert_index) → augmented_image."""


def _default_augment(image: torch.Tensor, expert_index: int) -> torch.Tensor:
    """Apply per-expert additive noise to encourage representational diversity.

    Each expert receives a differently perturbed view of the same sample.
    Expert 0 (typically global/PCA) sees clean data; others see increasing noise,
    which nudges their receptive fields towards different local features.
    """
    _noise_scales = [0.00, 0.04, 0.07, 0.05, 0.02]
    scale = _noise_scales[expert_index % len(_noise_scales)]
    if scale <= 0.0:
        return image
    return image.add(torch.randn_like(image).mul_(scale)).clamp_(0.0, 1.0)


@dataclass
class ExpertTrainMetrics:
    """Per-expert summary produced by :class:`EnsembleTrainer`."""

    name: str
    accuracy: float
    per_class_accuracy: dict[int, float]
    total_predictions: int


@dataclass
class EnsembleTrainMetrics:
    """Full ensemble training summary produced by :class:`EnsembleTrainer`."""

    total_samples: int
    expert_metrics: list[ExpertTrainMetrics]
    ensemble_accuracy: float
    disagreement_rate: float
    boosting_weight_stats: dict[str, float]  # keys: min, max, mean, std

    @property
    def best_expert(self) -> ExpertTrainMetrics | None:
        """Return the expert with the highest accuracy, or None if there are no experts."""
        return max(self.expert_metrics, key=lambda m: m.accuracy, default=None)


class EnsembleTrainer:
    """Online ensemble trainer that trains experts with per-expert augmentation.

    Wraps an :class:`~bioarn.ensemble.EnsemblePool` and runs one streaming pass
    through ``samples``, applying independently augmented views to each expert.
    Per-expert augmentation (additive noise at different scales by default)
    encourages the experts to specialise on different features even when they
    share the same architecture, adding a second source of diversity on top of
    the architectural diversity created by :class:`~bioarn.ensemble.DiversityManager`.

    Hebbian boosting weights are updated after every labelled sample via
    :meth:`~bioarn.ensemble.EnsemblePool.update_with_feedback`, so the boosting
    matrix reflects per-expert per-class performance throughout training.

    Two equivalent usage patterns are supported::

        # Pattern A — ensemble passed to train()
        trainer = EnsembleTrainer(num_classes=10)
        metrics = trainer.train(pool, samples)

        # Pattern B — ensemble bound at construction
        trainer = EnsembleTrainer(pool, num_classes=10)
        metrics = trainer.train(samples)
    """

    def __init__(
        self,
        ensemble: EnsemblePool | None = None,
        *,
        num_classes: int = 10,
        log_every: int = 200,
        augment_fn: AugmentFn | None = None,
        use_default_augmentation: bool = True,
    ) -> None:
        """
        Args:
            ensemble: Optional :class:`~bioarn.ensemble.EnsemblePool` to bind at
                construction (Pattern B).  When supplied, ``train()`` only requires
                ``samples`` as its first argument.
            num_classes: Number of target classes; controls ``per_class_accuracy``
                resolution in returned metrics.
            log_every: Print a progress line every N samples (0 = silent).
            augment_fn: Custom ``(image, expert_index) → image`` callable.  When
                supplied, ``use_default_augmentation`` is ignored.
            use_default_augmentation: Apply the built-in per-expert noise
                augmentation when no custom ``augment_fn`` is provided.
        """
        self._ensemble = ensemble
        self.num_classes = int(max(1, num_classes))
        self.log_every = max(0, int(log_every))
        self._augment_fn: AugmentFn | None = augment_fn
        self._use_default = bool(use_default_augmentation) and augment_fn is None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def train(
        self,
        ensemble_or_samples: "EnsemblePool | Iterable[tuple[torch.Tensor, int | None]]",
        samples: "Iterable[tuple[torch.Tensor, int | None]] | None" = None,
        *,
        num_classes: int | None = None,
    ) -> EnsembleTrainMetrics:
        """Train all experts online with per-expert augmentation.

        Supports two call signatures:

        * ``train(ensemble, samples)`` — ensemble passed explicitly each call.
        * ``train(samples)`` — ensemble was provided at construction.

        For each sample the trainer:

        1. Classifies using the original (clean) image to get fair per-expert
           predictions for metric tracking.
        2. Updates Hebbian boosting weights via
           :meth:`~bioarn.ensemble.EnsemblePool.update_with_feedback`.
        3. Runs one Hebbian learning step per expert on that expert's own
           augmented view of the image.

        Args:
            ensemble_or_samples: Either an :class:`~bioarn.ensemble.EnsemblePool`
                (Pattern A) or the samples iterable (Pattern B, when the pool was
                provided at construction).
            samples: Iterable of ``(flat_image, label)`` pairs when using Pattern A.
                ``label`` may be ``None`` for unsupervised samples.
            num_classes: Override the ``num_classes`` set at construction for
                per-class accuracy reporting.

        Returns:
            An :class:`EnsembleTrainMetrics` with per-expert accuracy, diversity,
            and boosting-weight statistics.
        """
        if isinstance(ensemble_or_samples, EnsemblePool):
            actual_ensemble: EnsemblePool = ensemble_or_samples
            if samples is None:
                raise ValueError("samples must be provided when ensemble is the first argument.")
            actual_samples: Iterable[tuple[torch.Tensor, int | None]] = samples
        else:
            if self._ensemble is None:
                raise ValueError(
                    "No ensemble provided at construction. "
                    "Either pass EnsemblePool as first argument or bind it at construction."
                )
            actual_ensemble = self._ensemble
            actual_samples = ensemble_or_samples  # type: ignore[assignment]

        return self._do_train(actual_ensemble, actual_samples, num_classes=num_classes)

    def _do_train(
        self,
        ensemble: EnsemblePool,
        samples: Iterable[tuple[torch.Tensor, int | None]],
        *,
        num_classes: int | None,
    ) -> EnsembleTrainMetrics:
        num_cls = int(num_classes) if num_classes is not None else self.num_classes
        sample_list = list(samples)
        if not sample_list:
            return self._empty_metrics(ensemble, num_cls)

        names = [state.name for state in ensemble.experts]
        expert_correct: dict[str, int] = {n: 0 for n in names}
        expert_total: dict[str, int] = {n: 0 for n in names}
        expert_class_correct: dict[str, Counter[int]] = {n: Counter() for n in names}
        expert_class_total: dict[str, Counter[int]] = {n: Counter() for n in names}
        # Prediction histories for diversity measurement; -1 marks abstentions.
        pred_histories: list[list[int]] = [[] for _ in ensemble.experts]
        ensemble_correct = 0
        ensemble_labeled = 0

        n_samples = len(sample_list)
        progress_interval = max(1, n_samples // 10)

        for step, (image, label) in enumerate(sample_list, start=1):
            if label is not None:
                # Classify on the clean image so metrics are uncontaminated by noise.
                result = ensemble.classify(image)

                for idx, (state, ep) in enumerate(
                    zip(ensemble.experts, result.expert_results, strict=False)
                ):
                    pred = -1 if ep.abstained else int(ep.predicted_class)
                    pred_histories[idx].append(pred)
                    expert_total[state.name] += 1
                    expert_class_total[state.name][int(label)] += 1
                    if not ep.abstained and ep.predicted_class == int(label):
                        expert_correct[state.name] += 1
                        expert_class_correct[state.name][int(label)] += 1

                if not result.abstained and result.predicted_class == int(label):
                    ensemble_correct += 1
                ensemble_labeled += 1

                # Update Hebbian boosting weights for each expert+class pair.
                ensemble.update_with_feedback(result, int(label))

            # Learn: each expert trains on its own augmented view.
            for idx, state in enumerate(ensemble.experts):
                aug = self._augment(image, idx)
                ensemble._learn_expert(state, aug, label)  # noqa: SLF001

            if self.log_every > 0 and (
                step % progress_interval == 0 or step == n_samples
            ):
                acc_str = " ".join(
                    f"{n}={expert_correct[n] / max(expert_total[n], 1):.3f}"
                    for n in names
                )
                ens_acc = ensemble_correct / max(ensemble_labeled, 1)
                print(
                    f"[ensemble_trainer] {step}/{n_samples}"
                    f" {acc_str} ensemble={ens_acc:.3f}"
                )

        disagreement = self._compute_diversity(pred_histories)
        expert_metrics = self._build_expert_metrics(
            ensemble, expert_correct, expert_total, expert_class_correct, expert_class_total, num_cls
        )
        weight_stats = self._boosting_weight_stats(ensemble)

        return EnsembleTrainMetrics(
            total_samples=n_samples,
            expert_metrics=expert_metrics,
            ensemble_accuracy=ensemble_correct / max(ensemble_labeled, 1),
            disagreement_rate=disagreement,
            boosting_weight_stats=weight_stats,
        )

    def report(self, metrics: EnsembleTrainMetrics) -> str:
        """Return a human-readable summary of the training metrics."""
        lines = [
            f"EnsembleTrainer — {metrics.total_samples} samples",
            f"  ensemble_accuracy : {metrics.ensemble_accuracy:.3f}",
            f"  disagreement_rate : {metrics.disagreement_rate:.3f}",
            "  expert accuracies :",
        ]
        for em in metrics.expert_metrics:
            lines.append(f"    {em.name:<24} {em.accuracy:.3f}")
        ws = metrics.boosting_weight_stats
        lines.append(
            f"  boosting_weights  : "
            f"min={ws['min']:.3f} max={ws['max']:.3f} "
            f"mean={ws['mean']:.3f} std={ws['std']:.3f}"
        )
        if metrics.best_expert is not None:
            lines.append(f"  best_expert       : {metrics.best_expert.name}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _augment(self, image: torch.Tensor, expert_index: int) -> torch.Tensor:
        if self._augment_fn is not None:
            return self._augment_fn(image, expert_index)
        if self._use_default:
            return _default_augment(image, expert_index)
        return image

    @staticmethod
    def _compute_diversity(pred_histories: list[list[int]]) -> float:
        if len(pred_histories) < 2:
            return 0.0
        min_len = min(len(h) for h in pred_histories)
        if min_len == 0:
            return 0.0
        trimmed = [h[:min_len] for h in pred_histories]
        return DiversityManager().measure_diversity(trimmed)

    @staticmethod
    def _build_expert_metrics(
        ensemble: EnsemblePool,
        expert_correct: dict[str, int],
        expert_total: dict[str, int],
        expert_class_correct: dict[str, Counter[int]],
        expert_class_total: dict[str, Counter[int]],
        num_cls: int,
    ) -> list[ExpertTrainMetrics]:
        result = []
        for state in ensemble.experts:
            per_class = {
                c: expert_class_correct[state.name].get(c, 0)
                / max(expert_class_total[state.name].get(c, 0), 1)
                for c in range(num_cls)
            }
            result.append(
                ExpertTrainMetrics(
                    name=state.name,
                    accuracy=expert_correct[state.name] / max(expert_total[state.name], 1),
                    per_class_accuracy=per_class,
                    total_predictions=expert_total[state.name],
                )
            )
        return result

    @staticmethod
    def _boosting_weight_stats(ensemble: EnsemblePool) -> dict[str, float]:
        if ensemble.boosting is None or ensemble.boosting.weights.numel() == 0:
            return {"min": 1.0, "max": 1.0, "mean": 1.0, "std": 0.0}
        w = ensemble.boosting.weights
        return {
            "min": float(w.min().item()),
            "max": float(w.max().item()),
            "mean": float(w.mean().item()),
            "std": float(w.std().item()) if w.numel() > 1 else 0.0,
        }

    @staticmethod
    def _empty_metrics(ensemble: EnsemblePool, num_cls: int) -> EnsembleTrainMetrics:
        empty_per_class = {c: 0.0 for c in range(num_cls)}
        expert_metrics = [
            ExpertTrainMetrics(
                name=state.name,
                accuracy=0.0,
                per_class_accuracy=empty_per_class,
                total_predictions=0,
            )
            for state in ensemble.experts
        ]
        return EnsembleTrainMetrics(
            total_samples=0,
            expert_metrics=expert_metrics,
            ensemble_accuracy=0.0,
            disagreement_rate=0.0,
            boosting_weight_stats={"min": 1.0, "max": 1.0, "mean": 1.0, "std": 0.0},
        )


__all__ = ["AugmentFn", "EnsembleTrainMetrics", "EnsembleTrainer", "ExpertTrainMetrics"]
