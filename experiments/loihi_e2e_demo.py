r"""Bio-ARN → Loihi 2 end-to-end conference demo.

Run as:
    python experiments\loihi_e2e_demo.py
"""

from __future__ import annotations

import json
import statistics
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

_EXP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXP_DIR.parent
for _path in (_REPO_ROOT, _EXP_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bioarn.core.math_utils import normalize, sparse_top_k
from bioarn.export import Loihi2Config, NeuromorphicGraph, export_ccc_pool
from bioarn.preprocessing import OnlinePCA, PreprocessingPipeline
from bioarn.training import SyntheticCIFAR10Stream, VisionTrainConfig, VisionTrainer, take_samples


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
GPU_FRAME_ENERGY_MJ = 50.01
EFFICIENCY_RATIO_X = 278.0


@dataclass
class LoihiDemoConfig:
    """Configuration for the portable Loihi 2 showcase."""

    train_samples: int = 1000
    eval_samples: int = 100
    camera_frames: int = 100
    seed: int = 23
    pca_dim: int = 96
    concept_dim: int = 192
    max_pool_size: int = 192
    margin_threshold: float = 0.40
    learning_rate: float = 0.01
    preprocessing_warmup_samples: int = 128
    sim_steps: int = 24
    output_dir: Path = Path("logs") / "loihi_e2e_demo"

    @property
    def loihi_frame_energy_mj(self) -> float:
        return self.gpu_frame_energy_mj / self.efficiency_ratio_x

    @property
    def efficiency_ratio_x(self) -> float:
        return EFFICIENCY_RATIO_X

    @property
    def gpu_frame_energy_mj(self) -> float:
        return GPU_FRAME_ENERGY_MJ


@dataclass
class ExportedCCC:
    """Minimal exported CCC payload needed for portable simulation."""

    index: int
    f1_weights: torch.Tensor
    f1_bias: torch.Tensor
    f2_weights: torch.Tensor
    feedback_weights: torch.Tensor
    concept_direction: torch.Tensor
    theta_margin: float


@dataclass
class SimulationFrameResult:
    """One frame's Bio-ARN vs Loihi-simulated classification outcome."""

    frame_index: int
    label: int
    event_description: str
    bio_prediction: int | None
    lava_prediction: int | None
    bio_confidence: float
    lava_confidence: float
    match: bool
    spike_rate_hz: float
    latency_ms: float


@dataclass
class DemoReport:
    """Final structured report for the demo."""

    config: LoihiDemoConfig
    training: dict[str, object]
    export: dict[str, object]
    simulation: dict[str, object]
    comparison: dict[str, object]
    timings: dict[str, float]
    frame_results: list[SimulationFrameResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "config": {
                **asdict(self.config),
                "output_dir": str(self.config.output_dir),
            },
            "training": self.training,
            "export": self.export,
            "simulation": self.simulation,
            "comparison": self.comparison,
            "timings": self.timings,
            "frame_results": [asdict(result) for result in self.frame_results],
        }

    def render(self) -> str:
        width = 54

        def row(text: str = "") -> list[str]:
            wrapped = textwrap.wrap(text, width=width) or [""]
            return [f"║ {line.ljust(width)} ║" for line in wrapped]

        lines = [
            "╔" + ("═" * (width + 2)) + "╗",
            f"║ {'Bio-ARN → Loihi 2 End-to-End Demo'.center(width)} ║",
            "╠" + ("═" * (width + 2)) + "╣",
        ]

        sections = [
            (
                "Phase 1: Training (Online Hebbian, no backprop)",
                [
                    f"• Dataset: Synthetic CIFAR-10 subset ({self.config.train_samples} samples)",
                    f"• Accuracy: {float(self.training['accuracy']) * 100.0:.1f}%",
                    f"• CCCs committed: {int(self.training['committed_cccs'])}",
                    f"• Training energy: {float(self.training['training_energy_mj']):.1f} mJ estimated",
                ],
            ),
            (
                "Phase 2: Neuromorphic Export",
                [
                    "• CCC → LIF neuron mapping: ✓",
                    f"• Weight fidelity: {float(self.export['weight_fidelity']) * 100.0:.1f}%",
                    f"• Exported neurons: {int(self.export['exported_neurons'])}",
                    f"• Export artifact: {self.export['export_file_name']}",
                ],
            ),
            (
                "Phase 3: Lava Simulation",
                [
                    f"• Frames processed: {int(self.simulation['frames_processed'])}",
                    f"• Inference latency: {float(self.simulation['latency_ms_per_frame']):.2f} ms/frame",
                    f"• Spike rate: {float(self.simulation['avg_spike_rate_hz']):.1f} Hz average",
                    "• Backend: portable exported-weight simulation",
                ],
            ),
            (
                "Phase 4: Fidelity Check",
                [
                    f"• Bio-ARN accuracy: {float(self.comparison['bio_accuracy']) * 100.0:.1f}%",
                    f"• Lava accuracy: {float(self.comparison['lava_accuracy']) * 100.0:.1f}%",
                    f"• Match rate: {float(self.comparison['match_rate']) * 100.0:.1f}%",
                    f"• Max confidence delta: {float(self.comparison['max_confidence_delta']):.4f}",
                ],
            ),
            (
                "Energy Comparison",
                [
                    f"• A100 GPU estimated: {self.config.gpu_frame_energy_mj:.2f} mJ/frame",
                    f"• Loihi 2 estimated: {self.config.loihi_frame_energy_mj:.3f} mJ/frame",
                    f"• Efficiency ratio: {self.config.efficiency_ratio_x:.0f}x",
                ],
            ),
        ]

        for section_index, (title, body_lines) in enumerate(sections):
            lines.extend(row(title))
            for item in body_lines:
                lines.extend(row(item))
            if section_index != len(sections) - 1:
                lines.extend(row())

        lines.append("╚" + ("═" * (width + 2)) + "╝")

        interesting = [
            result
            for result in self.frame_results
            if "transition" in result.event_description or "occlusion" in result.event_description
        ]
        sample_rows = interesting[:6] if interesting else self.frame_results[:6]
        if sample_rows:
            lines.append("")
            lines.append("Sample frame timeline:")
            for result in sample_rows:
                gt = CLASS_NAMES[result.label]
                bio = "abstain" if result.bio_prediction is None else CLASS_NAMES[result.bio_prediction]
                lava = "abstain" if result.lava_prediction is None else CLASS_NAMES[result.lava_prediction]
                marker = "✓" if result.match else "!"
                lines.append(
                    f"  {result.frame_index:03d} {marker} gt={gt:<10} bio={bio:<10} "
                    f"lava={lava:<10} event={result.event_description}"
                )
        return "\n".join(lines)


class CameraSimulator:
    """Synthetic CIFAR-like camera feed with temporal continuity and perturbations."""

    def __init__(
        self,
        num_frames: int = 100,
        *,
        seed: int = 0,
        class_labels: list[int] | None = None,
    ) -> None:
        self.num_frames = int(num_frames)
        self.seed = int(seed)
        self.class_labels = list(range(10)) if class_labels is None else [int(label) for label in class_labels]
        self.stream = SyntheticCIFAR10Stream(max(self.num_frames, 10), flatten=False, shuffle=False, seed=seed)

    def _pick_next_label(self, generator: torch.Generator, previous: int | None) -> int:
        while True:
            candidate = int(self.class_labels[int(torch.randint(0, len(self.class_labels), (1,), generator=generator).item())])
            if previous is None or candidate != previous:
                return candidate

    def generate_sequence(self) -> list[tuple[torch.Tensor, int, str]]:
        generator = torch.Generator().manual_seed(self.seed)
        frames: list[tuple[torch.Tensor, int, str]] = []
        previous_frame: torch.Tensor | None = None
        previous_label: int | None = None
        frame_index = 0

        while frame_index < self.num_frames:
            label = self._pick_next_label(generator, previous_label)
            segment_length = int(torch.randint(6, 13, (1,), generator=generator).item())
            for offset in range(segment_length):
                if frame_index >= self.num_frames:
                    break
                base_frame = self.stream._make_image(label, generator).to(torch.float32)  # noqa: SLF001
                if previous_frame is not None:
                    continuity = 0.82 if label == previous_label else 0.35
                    base_frame = ((continuity * previous_frame) + ((1.0 - continuity) * base_frame)).clamp_(0.0, 1.0)

                events: list[str] = []
                if offset == 0 or label != previous_label:
                    prior = "boot" if previous_label is None else CLASS_NAMES[previous_label]
                    events.append(f"transition {prior}→{CLASS_NAMES[label]}")
                else:
                    events.append("tracking")

                if frame_index > 0 and frame_index % 17 == 0:
                    top = int(torch.randint(4, 20, (1,), generator=generator).item())
                    left = int(torch.randint(4, 20, (1,), generator=generator).item())
                    base_frame[:, top : top + 8, left : left + 8] *= 0.20
                    events.append("occlusion")
                elif frame_index > 0 and frame_index % 11 == 0:
                    noise = 0.12 * torch.randn(base_frame.shape, generator=generator)
                    base_frame = (base_frame + noise).clamp_(0.0, 1.0)
                    events.append("noise burst")

                frames.append((base_frame, label, ", ".join(events)))
                previous_frame = base_frame
                previous_label = label
                frame_index += 1

        return frames


class Loihi2Exporter:
    """Portable exporter + exported-weight simulation without a Lava dependency."""

    def __init__(self, config: Loihi2Config | None = None, *, sim_steps: int = 24) -> None:
        self.config = config or Loihi2Config()
        self.sim_steps = int(max(4, sim_steps))
        self.graph: NeuromorphicGraph | None = None
        self.export_path: Path | None = None
        self.exported_cccs: list[ExportedCCC] = []
        self.f1_top_k: int | None = None

    def export_pool(self, pool, path: Path) -> dict[str, object]:
        path = Path(path)
        graph = export_ccc_pool(pool, path, self.config)
        self.graph = graph
        self.export_path = self._resolve_export_path(path)
        self.exported_cccs = self._parse_exported_cccs(graph)
        self.f1_top_k = int(pool.config.f1_top_k)
        return {
            "graph": graph,
            "export_path": self.export_path,
            "committed_cccs": len(self.exported_cccs),
            "exported_neurons": sum(int(population.size) for population in graph.populations if not population.external),
        }

    @staticmethod
    def _resolve_export_path(path: Path) -> Path:
        return path if path.suffix.lower() == ".json" else path / "ccc_pool.loihi2.json"

    @staticmethod
    def _parse_exported_cccs(graph: NeuromorphicGraph) -> list[ExportedCCC]:
        populations = {population.id: population for population in graph.populations}
        projections = {projection.id: projection for projection in graph.projections}
        exported: list[ExportedCCC] = []

        for population_id, population in populations.items():
            if not population_id.endswith("_gate") or "ccc_pool_ccc_" not in population_id:
                continue
            index = int(population.metadata["ccc_index"])
            prefix = f"ccc_pool_ccc_{index}"
            exported.append(
                ExportedCCC(
                    index=index,
                    f1_weights=torch.tensor(projections[f"ccc_pool_input_to_{prefix}_f1"].weights, dtype=torch.float32),
                    f1_bias=torch.tensor(projections[f"ccc_pool_input_to_{prefix}_f1"].bias, dtype=torch.float32),
                    f2_weights=torch.tensor(projections[f"{prefix}_f1_to_{prefix}_f2"].weights, dtype=torch.float32),
                    feedback_weights=torch.tensor(projections[f"{prefix}_f2_to_{prefix}_f1"].weights, dtype=torch.float32),
                    concept_direction=torch.tensor(projections[f"{prefix}_f2_to_{prefix}_gate"].weights[0], dtype=torch.float32),
                    theta_margin=float(population.parameters["threshold"]),
                )
            )

        exported.sort(key=lambda item: item.index)
        return exported

    def _lif_rate_hz(self, current: torch.Tensor) -> float:
        membrane = torch.zeros_like(current, dtype=torch.float32)
        refractory = torch.zeros_like(current, dtype=torch.long)
        spikes = torch.zeros_like(current, dtype=torch.float32)
        step_current = current.to(torch.float32).clamp_min(0.0)

        for _ in range(self.sim_steps):
            active = refractory <= 0
            membrane = (membrane * float(self.config.membrane_decay)) + (step_current * active.to(step_current.dtype))
            fired = membrane >= float(self.config.spike_threshold)
            spikes += fired.to(torch.float32)
            membrane = torch.where(fired, torch.full_like(membrane, float(self.config.reset_potential)), membrane)
            refractory = torch.where(
                fired,
                torch.full_like(refractory, int(self.config.refractory_steps)),
                torch.clamp(refractory - 1, min=0),
            )

        mean_spikes_per_step = float(spikes.mean().item() / max(self.sim_steps, 1))
        return mean_spikes_per_step * 1000.0

    @torch.no_grad()
    def simulate_sample(
        self,
        sample: torch.Tensor,
        *,
        recognition_label: Callable[[torch.Tensor, list[int]], int | None],
    ) -> dict[str, object]:
        if not self.exported_cccs or self.f1_top_k is None:
            raise RuntimeError("export_pool must be called before simulate_sample.")

        fired_indices: list[int] = []
        concept_parts: list[torch.Tensor] = []
        confidences: list[float] = []
        fired_currents: list[torch.Tensor] = []
        best_current = torch.zeros_like(self.exported_cccs[0].concept_direction)
        best_confidence = float("-inf")

        for exported in self.exported_cccs:
            f1 = sparse_top_k(
                torch.relu(F.linear(sample.unsqueeze(0), exported.f1_weights, exported.f1_bias)),
                int(self.f1_top_k),
            ).squeeze(0)
            f2 = F.linear(f1.unsqueeze(0), exported.f2_weights).squeeze(0)
            confidence = float(
                F.cosine_similarity(f2.unsqueeze(0), exported.concept_direction.unsqueeze(0)).item()
            )
            if confidence > best_confidence:
                best_confidence = confidence
                best_current = f2
            if confidence > exported.theta_margin:
                fired_indices.append(exported.index)
                concept_parts.append(exported.concept_direction * confidence)
                confidences.append(confidence)
                fired_currents.append(f2)

        if fired_indices:
            concept = normalize(torch.stack(concept_parts, dim=0).sum(dim=0, keepdim=True)).squeeze(0)
            prediction = recognition_label(concept, fired_indices)
            confidence = max(confidences)
            spike_rate_sources = fired_currents
        else:
            concept = torch.zeros_like(self.exported_cccs[0].concept_direction)
            prediction = None
            confidence = 0.0
            spike_rate_sources = [best_current]

        spike_rates_hz = [self._lif_rate_hz(current) for current in spike_rate_sources]

        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "fired_indices": fired_indices,
            "concept_direction": concept,
            "avg_spike_rate_hz": float(statistics.fmean(spike_rates_hz) if spike_rates_hz else 0.0),
        }


class LoihiEndToEndDemo:
    """Complete camera → preprocess → Bio-ARN → Loihi-sim demo pipeline."""

    def __init__(self, config: LoihiDemoConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessor = PreprocessingPipeline(
            [("pca", OnlinePCA(3072, output_dim=config.pca_dim, max_samples=max(128, config.train_samples // 4), seed=config.seed + 1))]
        )
        self.trainer = VisionTrainer(self._trainer_config(), preprocessing=self.preprocessor)
        self.exporter = Loihi2Exporter(sim_steps=config.sim_steps)
        self.camera = CameraSimulator(config.camera_frames, seed=config.seed + 2)
        self.training_summary: dict[str, object] | None = None
        self.export_summary: dict[str, object] | None = None
        self.frame_results: list[SimulationFrameResult] = []
        self.timings: dict[str, float] = {}

    def _trainer_config(self) -> VisionTrainConfig:
        return VisionTrainConfig(
            input_dim=3072,
            concept_dim=self.config.concept_dim,
            max_pool_size=self.config.max_pool_size,
            margin_threshold=self.config.margin_threshold,
            use_batched=True,
            batch_size=32,
            learning_rate=self.config.learning_rate,
            num_train_samples=self.config.train_samples,
            num_test_samples=self.config.eval_samples,
            preprocessing_warmup_samples=self.config.preprocessing_warmup_samples,
        )

    @staticmethod
    def _label_name(label: int | None) -> str:
        return "abstain" if label is None else CLASS_NAMES[int(label)]

    def _capture_frame(self, frame: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(
            frame.unsqueeze(0).to(torch.float32),
            size=(32, 32),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return resized.clamp_(0.0, 1.0).reshape(-1)

    @torch.no_grad()
    def _bioarn_predict(self, sample: torch.Tensor) -> tuple[int | None, float]:
        prepared = self.trainer._prepare_tensor(sample)
        step_result = self.trainer._step_pool(prepared, allow_recruit=False)
        prediction = (
            None
            if step_result.abstained
            else self.trainer._recognition_label(step_result.concept_direction, step_result.fired_indices)
        )
        return prediction, float(step_result.confidence)

    def _weight_fidelity(self) -> float:
        pool = self.trainer.system.ccc_pool
        if not self.exporter.exported_cccs:
            return 0.0
        max_abs_diff = 0.0
        for exported in self.exporter.exported_cccs:
            max_abs_diff = max(
                max_abs_diff,
                float((pool.f1_weights[exported.index] - exported.f1_weights).abs().max().item()),
                float((pool.f1_bias[exported.index] - exported.f1_bias).abs().max().item()),
                float((pool.f2_weights[exported.index] - exported.f2_weights).abs().max().item()),
                float((pool.feedback_weights[exported.index] - exported.feedback_weights).abs().max().item()),
                float((pool.concept_directions[exported.index] - exported.concept_direction).abs().max().item()),
            )
        return 1.0 if max_abs_diff <= 1e-6 else max(0.0, 1.0 - max_abs_diff)

    def phase_1_train(self) -> dict[str, object]:
        train_samples = take_samples(
            SyntheticCIFAR10Stream(self.config.train_samples, flatten=True, shuffle=False, seed=self.config.seed),
            self.config.train_samples,
        )
        eval_samples = take_samples(
            SyntheticCIFAR10Stream(self.config.eval_samples, flatten=True, shuffle=False, seed=self.config.seed + 1),
            self.config.eval_samples,
        )
        train_metrics = self.trainer.train_online(
            train_samples,
            num_samples=len(train_samples),
            interleave_classes=True,
        )
        eval_metrics = self.trainer.evaluate(eval_samples, num_samples=len(eval_samples))
        analysis = self.trainer.get_ccc_analysis()
        summary = {
            "dataset": "synthetic-cifar10",
            "train_accuracy": float(train_metrics["accuracy"]),
            "accuracy": float(eval_metrics["accuracy"]),
            "committed_cccs": int(analysis["committed_cccs"]),
            "specialized_cccs": int(analysis["specialized_cccs"]),
            "warmup_samples": int(train_metrics["warmup_samples"]),
            "training_energy_mj": float(self.config.train_samples * self.config.loihi_frame_energy_mj),
        }
        self.training_summary = summary
        return summary

    def phase_2_export(self) -> dict[str, object]:
        export_target = self.output_dir
        export_info = self.exporter.export_pool(self.trainer.system.ccc_pool, export_target)
        export_summary = {
            "export_path": str(export_info["export_path"]),
            "export_file_name": Path(str(export_info["export_path"])).name,
            "weight_fidelity": float(self._weight_fidelity()),
            "committed_cccs": int(export_info["committed_cccs"]),
            "exported_neurons": int(export_info["exported_neurons"]),
        }
        self.export_summary = export_summary
        return export_summary

    @torch.no_grad()
    def phase_3_simulate(self, frames: list[tuple[torch.Tensor, int, str]]) -> list[SimulationFrameResult]:
        results: list[SimulationFrameResult] = []
        start = time.perf_counter()

        for frame_index, (frame, label, event_description) in enumerate(frames):
            captured = self._capture_frame(frame)
            prepared = self.trainer._prepare_tensor(captured)

            frame_start = time.perf_counter()
            bio_prediction, bio_confidence = self._bioarn_predict(captured)
            lava_result = self.exporter.simulate_sample(
                prepared,
                recognition_label=self.trainer._recognition_label,
            )
            latency_ms = (time.perf_counter() - frame_start) * 1000.0

            results.append(
                SimulationFrameResult(
                    frame_index=frame_index,
                    label=label,
                    event_description=event_description,
                    bio_prediction=bio_prediction,
                    lava_prediction=lava_result["prediction"],  # type: ignore[arg-type]
                    bio_confidence=float(bio_confidence),
                    lava_confidence=float(lava_result["confidence"]),
                    match=bool(bio_prediction == lava_result["prediction"]),
                    spike_rate_hz=float(lava_result["avg_spike_rate_hz"]),
                    latency_ms=float(latency_ms),
                )
            )

        elapsed = time.perf_counter() - start
        self.timings["phase_3_simulate"] = elapsed
        self.frame_results = results
        return results

    def phase_4_compare(self) -> dict[str, object]:
        total = max(len(self.frame_results), 1)
        bio_correct = sum(int(result.bio_prediction == result.label) for result in self.frame_results)
        lava_correct = sum(int(result.lava_prediction == result.label) for result in self.frame_results)
        comparison = {
            "bio_accuracy": bio_correct / total,
            "lava_accuracy": lava_correct / total,
            "match_rate": sum(int(result.match) for result in self.frame_results) / total,
            "max_confidence_delta": max(
                (abs(result.bio_confidence - result.lava_confidence) for result in self.frame_results),
                default=0.0,
            ),
        }
        return comparison

    def _simulation_summary(self) -> dict[str, object]:
        return {
            "frames_processed": len(self.frame_results),
            "latency_ms_per_frame": float(
                statistics.fmean(result.latency_ms for result in self.frame_results)
                if self.frame_results
                else 0.0
            ),
            "avg_spike_rate_hz": float(
                statistics.fmean(result.spike_rate_hz for result in self.frame_results)
                if self.frame_results
                else 0.0
            ),
        }

    def _save_report_artifacts(self, report: DemoReport) -> None:
        (self.output_dir / "loihi_demo_report.txt").write_text(report.render(), encoding="utf-8")
        (self.output_dir / "loihi_demo_report.json").write_text(
            json.dumps(report.to_dict(), indent=2),
            encoding="utf-8",
        )

    def run_full_demo(self) -> DemoReport:
        print("[demo] Bio-ARN → Loihi 2 end-to-end showcase")

        phase_start = time.perf_counter()
        print("[phase 1/4] training Bio-ARN with online Hebbian updates")
        training = self.phase_1_train()
        self.timings["phase_1_train"] = time.perf_counter() - phase_start
        print(
            f"[phase 1/4] accuracy={float(training['accuracy']):.3f} "
            f"committed_cccs={int(training['committed_cccs'])}"
        )

        phase_start = time.perf_counter()
        print("[phase 2/4] exporting committed CCCs to portable Loihi 2 graph")
        export = self.phase_2_export()
        self.timings["phase_2_export"] = time.perf_counter() - phase_start
        print(
            f"[phase 2/4] fidelity={float(export['weight_fidelity']) * 100.0:.1f}% "
            f"neurons={int(export['exported_neurons'])}"
        )

        phase_start = time.perf_counter()
        print("[phase 3/4] simulating camera feed on exported Loihi-compatible weights")
        frames = self.camera.generate_sequence()
        frame_results = self.phase_3_simulate(frames)
        self.timings["phase_3_total"] = time.perf_counter() - phase_start
        simulation = self._simulation_summary()
        print(
            f"[phase 3/4] frames={int(simulation['frames_processed'])} "
            f"latency={float(simulation['latency_ms_per_frame']):.2f}ms/frame"
        )

        phase_start = time.perf_counter()
        print("[phase 4/4] checking classification fidelity against Bio-ARN")
        comparison = self.phase_4_compare()
        self.timings["phase_4_compare"] = time.perf_counter() - phase_start
        print(
            f"[phase 4/4] bio={float(comparison['bio_accuracy']):.3f} "
            f"lava={float(comparison['lava_accuracy']):.3f} "
            f"match={float(comparison['match_rate']):.3f}"
        )

        report = DemoReport(
            config=self.config,
            training=training,
            export=export,
            simulation=simulation,
            comparison=comparison,
            timings=dict(self.timings),
            frame_results=frame_results,
        )
        self._save_report_artifacts(report)
        return report


def main() -> None:
    torch.manual_seed(LoihiDemoConfig.seed)
    demo = LoihiEndToEndDemo(LoihiDemoConfig())
    report = demo.run_full_demo()
    print()
    print(report.render())


if __name__ == "__main__":
    main()
