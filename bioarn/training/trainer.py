"""Production online training loop for Bio-ARN."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

from bioarn.loop import SensorimotorLoop
from bioarn.system import BioARNCore, RecognitionOutput
from bioarn.utils.checkpoint import CheckpointManager
from bioarn.utils.config_manager import ConfigManager
from bioarn.utils.logging import BioARNLogger

Callback = Callable[..., None]


@dataclass
class EvalResult:
    """Evaluation summary for a trained Bio-ARN system."""

    accuracy: float
    abstention_rate: float
    sparsity: float
    latency_ms: float
    mean_free_energy: float
    total_samples: int


@dataclass
class TrainResult:
    """Training summary for an online Bio-ARN run."""

    total_steps: int
    accuracy: float
    abstention_rate: float
    ccc_recruitment_rate: float
    mean_free_energy: float
    mean_latency_ms: float
    checkpoints: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class _PrototypeBank:
    def __init__(self) -> None:
        self.prototypes: dict[int, torch.Tensor] = {}
        self.counts: dict[int, int] = {}

    def predict(self, concept: torch.Tensor) -> int | None:
        if not self.prototypes:
            return None
        labels = list(self.prototypes.keys())
        stacked = torch.stack([self.prototypes[label].to(concept) for label in labels], dim=0)
        query = concept.unsqueeze(0).expand_as(stacked)
        similarities = F.cosine_similarity(stacked, query, dim=-1)
        return labels[int(torch.argmax(similarities).item())]

    def update(self, label: int, concept: torch.Tensor) -> None:
        count = self.counts.get(label, 0) + 1
        if label not in self.prototypes:
            self.prototypes[label] = concept.detach().clone()
        else:
            momentum = (count - 1) / count
            self.prototypes[label] = F.normalize(
                (self.prototypes[label] * momentum) + (concept.detach().clone() / count),
                dim=0,
            )
        self.counts[label] = count

    def to_state(self) -> dict[str, Any]:
        return {
            str(label): {
                "count": self.counts[label],
                "prototype": prototype.detach().clone(),
            }
            for label, prototype in self.prototypes.items()
        }

    @classmethod
    def from_state(cls, state: dict[str, Any] | None) -> "_PrototypeBank":
        bank = cls()
        for label, payload in (state or {}).items():
            bank.prototypes[int(label)] = payload["prototype"].detach().clone()
            bank.counts[int(label)] = int(payload["count"])
        return bank


class OnlineTrainer:
    """Production training loop for Bio-ARN (streaming, one-pass, online learning)."""

    def __init__(
        self,
        *,
        logger: BioARNLogger | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        log_every: int = 100,
        checkpoint_every: int = 1000,
        output_dir: str | Path | None = None,
        keep_last: int = 5,
    ) -> None:
        self.logger = logger or BioARNLogger(component="trainer")
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(keep_last=keep_last)
        self.log_every = max(1, int(log_every))
        self.checkpoint_every = max(1, int(checkpoint_every))
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.keep_last = max(1, int(keep_last))

    def train(
        self,
        system: BioARNCore | SensorimotorLoop,
        data_source: Iterable[Any],
        config: Any,
        callbacks: dict[str, Callback] | None = None,
    ) -> TrainResult:
        callbacks = callbacks or {}
        prototype_bank = _PrototypeBank()
        checkpoints: list[str] = []
        total_correct = 0
        total_labeled = 0
        total_abstained = 0
        total_recruited = 0
        free_energy_total = 0.0
        latency_total = 0.0
        total_steps = 0

        auto_checkpoint = None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            ConfigManager.to_yaml(ConfigManager.from_dict(config), self.output_dir / "resolved-config.yaml")
            auto_checkpoint = self.checkpoint_manager.auto_checkpoint(
                system,
                self.output_dir,
                interval_steps=self.checkpoint_every,
                keep_last=self.keep_last,
            )

        try:
            context = auto_checkpoint if auto_checkpoint is not None else _NullContext()
            with context as checkpoint_context:
                for total_steps, item in enumerate(data_source, start=1):
                    step_metrics = self._train_step(system, item, prototype_bank)
                    total_abstained += int(step_metrics["abstained"])
                    total_recruited += int(step_metrics["ccc_recruited"])
                    free_energy_total += float(step_metrics["free_energy"])
                    latency_total += float(step_metrics["latency_ms"])

                    label = step_metrics["label"]
                    if label is not None:
                        total_labeled += 1
                        total_correct += int(step_metrics["correct"])

                    callbacks.get("on_step", _noop)(total_steps, step_metrics)
                    if step_metrics["novel"]:
                        callbacks.get("on_novel_input", _noop)(total_steps, step_metrics, item)

                    if total_steps % self.log_every == 0:
                        snapshot = self._snapshot(
                            total_steps,
                            total_correct,
                            total_labeled,
                            total_abstained,
                            total_recruited,
                            free_energy_total,
                            latency_total,
                            system,
                        )
                        self.logger.log_metric("trainer", "accuracy", snapshot["accuracy"], total_steps)
                        self.logger.log_metric("trainer", "abstention_rate", snapshot["abstention_rate"], total_steps)
                        self.logger.log_metric("trainer", "free_energy", snapshot["mean_free_energy"], total_steps)
                        self.logger.log_event("trainer", "progress", snapshot)

                    if checkpoint_context is not None:
                        snapshot = self._snapshot(
                            total_steps,
                            total_correct,
                            total_labeled,
                            total_abstained,
                            total_recruited,
                            free_energy_total,
                            latency_total,
                            system,
                        )
                        checkpoint_path = checkpoint_context.maybe_save(
                            total_steps,
                            metadata={
                                "training_step": total_steps,
                                "metrics": snapshot,
                                "label_prototypes": prototype_bank.to_state(),
                            },
                        )
                        if checkpoint_path is not None:
                            checkpoints.append(str(checkpoint_path))
                            callbacks.get("on_checkpoint", _noop)(checkpoint_path, total_steps, snapshot)
        except KeyboardInterrupt:
            if auto_checkpoint is not None:
                interrupt_path = auto_checkpoint.save_now(
                    step=total_steps,
                    metadata={"training_step": total_steps, "interrupted": True},
                )
                checkpoints.append(str(interrupt_path))
            self.logger.warning("trainer", "interrupted", {"step": total_steps})

        final_metrics = self._snapshot(
            total_steps,
            total_correct,
            total_labeled,
            total_abstained,
            total_recruited,
            free_energy_total,
            latency_total,
            system,
        )
        final_metrics["label_prototypes"] = prototype_bank.to_state()

        if self.output_dir is not None:
            latest = self.checkpoint_manager.save(
                system,
                self.output_dir / "latest.pt",
                metadata={
                    "training_step": total_steps,
                    "metrics": final_metrics,
                    "label_prototypes": prototype_bank.to_state(),
                },
            )
            checkpoints.append(str(latest))

        return TrainResult(
            total_steps=total_steps,
            accuracy=float(final_metrics["accuracy"]),
            abstention_rate=float(final_metrics["abstention_rate"]),
            ccc_recruitment_rate=float(final_metrics["ccc_recruitment_rate"]),
            mean_free_energy=float(final_metrics["mean_free_energy"]),
            mean_latency_ms=float(final_metrics["mean_latency_ms"]),
            checkpoints=checkpoints,
            metrics=final_metrics,
        )

    def evaluate(
        self,
        system: BioARNCore | SensorimotorLoop,
        data_source: Iterable[Any],
        *,
        label_prototypes: dict[str, Any] | None = None,
    ) -> EvalResult:
        prototype_bank = _PrototypeBank.from_state(label_prototypes)
        total_correct = 0
        total_labeled = 0
        total_abstained = 0
        total_samples = 0
        free_energy_total = 0.0
        latency_total = 0.0
        sparsity_total = 0.0

        for item in data_source:
            input_tensor, label = self._normalize_item(system, item)
            total_samples += 1
            start = time.perf_counter()
            recognition, free_energy = self._inference(system, input_tensor)
            latency_total += (time.perf_counter() - start) * 1000.0
            free_energy_total += free_energy
            total_abstained += int(recognition.abstained)
            sparsity_total += float(self._system_stats(system)["sparsity"])

            if label is not None:
                total_labeled += 1
                if not recognition.abstained:
                    predicted = prototype_bank.predict(recognition.concept_direction)
                    total_correct += int(predicted == label)
                    if predicted is None:
                        prototype_bank.update(label, recognition.concept_direction)

        return EvalResult(
            accuracy=(total_correct / total_labeled) if total_labeled else 0.0,
            abstention_rate=(total_abstained / total_samples) if total_samples else 0.0,
            sparsity=(sparsity_total / total_samples) if total_samples else 0.0,
            latency_ms=(latency_total / total_samples) if total_samples else 0.0,
            mean_free_energy=(free_energy_total / total_samples) if total_samples else 0.0,
            total_samples=total_samples,
        )

    def _train_step(
        self,
        system: BioARNCore | SensorimotorLoop,
        item: Any,
        prototype_bank: _PrototypeBank,
    ) -> dict[str, Any]:
        input_tensor, label = self._normalize_item(system, item)
        start = time.perf_counter()
        concepts_before = int(self._system_stats(system)["concepts_learned"])

        if isinstance(system, SensorimotorLoop):
            output = self._loop_step(system, input_tensor)
            recognition = output.recognition
            free_energy = float(output.prediction.free_energy)
            novel = bool(output.reward.novelty.is_novel)
        else:
            output = system.forward(input_tensor, learn=True)
            recognition = RecognitionOutput(
                concept_direction=output.perception.vote_result.winning_direction.detach().clone(),
                confidence=float(output.perception.vote_result.confidence),
                abstained=bool(output.perception.vote_result.voter_count == 0),
                num_hypotheses=int(output.perception.num_fired),
                agreement=float(output.perception.vote_result.agreement_score),
            )
            free_energy = 0.0
            novel = bool(output.perception.is_novel)

        latency_ms = (time.perf_counter() - start) * 1000.0
        concepts_after = int(self._system_stats(system)["concepts_learned"])
        ccc_recruited = max(0, concepts_after - concepts_before)

        predicted: int | None = None
        correct = False
        if label is not None and not recognition.abstained:
            predicted = prototype_bank.predict(recognition.concept_direction)
            correct = predicted == label
            prototype_bank.update(label, recognition.concept_direction)

        return {
            "label": label,
            "predicted": predicted,
            "correct": correct,
            "abstained": recognition.abstained,
            "novel": novel,
            "ccc_recruited": ccc_recruited,
            "free_energy": free_energy,
            "latency_ms": latency_ms,
        }

    def _snapshot(
        self,
        total_steps: int,
        total_correct: int,
        total_labeled: int,
        total_abstained: int,
        total_recruited: int,
        free_energy_total: float,
        latency_total: float,
        system: BioARNCore | SensorimotorLoop,
    ) -> dict[str, Any]:
        stats = self._system_stats(system)
        accuracy = (total_correct / total_labeled) if total_labeled else 0.0
        return {
            "step": total_steps,
            "accuracy": accuracy,
            "abstention_rate": (total_abstained / total_steps) if total_steps else 0.0,
            "ccc_recruitment_rate": (total_recruited / total_steps) if total_steps else 0.0,
            "mean_free_energy": (free_energy_total / total_steps) if total_steps else 0.0,
            "mean_latency_ms": (latency_total / total_steps) if total_steps else 0.0,
            "concepts_learned": int(stats["concepts_learned"]),
            "sparsity": float(stats["sparsity"]),
        }

    @staticmethod
    def _system_stats(system: BioARNCore | SensorimotorLoop) -> dict[str, Any]:
        if isinstance(system, SensorimotorLoop):
            return system.core.get_system_stats()
        return system.get_system_stats()

    @staticmethod
    def _loop_step(system: SensorimotorLoop, input_tensor: torch.Tensor):
        if input_tensor.dtype in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.long,
            torch.uint8,
        }:
            return system.step(language_input=input_tensor)
        return system.step(visual_input=input_tensor)

    @staticmethod
    def _inference(system: BioARNCore | SensorimotorLoop, input_tensor: torch.Tensor) -> tuple[RecognitionOutput, float]:
        if isinstance(system, SensorimotorLoop):
            if input_tensor.dtype in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.long,
                torch.uint8,
            }:
                sensory = system.sense(language_input=input_tensor)
            else:
                sensory = system.sense(visual_input=input_tensor)
            prediction = system.predict(sensory.features)
            recognition = system.core.recognize(system._align_dim(sensory.features, system.sensory_dim))
            return recognition, float(prediction.free_energy)
        return system.recognize(input_tensor), 0.0

    @staticmethod
    def _normalize_item(system: BioARNCore | SensorimotorLoop, item: Any) -> tuple[torch.Tensor, int | None]:
        label: int | None = None
        payload = item
        if isinstance(item, dict):
            payload = item.get("input")
            label = int(item["label"]) if item.get("label") is not None else None
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            payload, label = item
            label = int(label) if label is not None else None

        if not isinstance(payload, torch.Tensor):
            payload = torch.as_tensor(payload)

        if isinstance(system, SensorimotorLoop):
            if payload.dtype in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.long,
                torch.uint8,
            }:
                return payload.reshape(-1).to(torch.long), label
            return payload.to(torch.float32).reshape(-1), label

        return payload.to(torch.float32).reshape(-1), label


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _noop(*args: Any, **kwargs: Any) -> None:
    del args, kwargs


__all__ = ["EvalResult", "OnlineTrainer", "TrainResult"]
