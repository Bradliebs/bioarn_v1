from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from bioarn.core.math_utils import cosine_similarity, normalize
from bioarn.loop import SensorimotorLoop
from bioarn.scaling import ScaledBioARN
from bioarn.system import BioARNCore, ContinualLearningEvaluator
from bioarn.training import OnlineTrainer
from bioarn.utils.checkpoint import CheckpointManager


def _build_label_prototypes(
    core: BioARNCore,
    samples: list[tuple[torch.Tensor, int]],
) -> dict[int, torch.Tensor]:
    grouped: dict[int, list[torch.Tensor]] = {}
    for vector, label in samples:
        recognition = core.recognize(vector)
        if recognition.abstained:
            continue
        grouped.setdefault(label, []).append(recognition.concept_direction.detach().clone())

    return {
        label: normalize(torch.stack(vectors, dim=0).mean(dim=0, keepdim=True)).squeeze(0)
        for label, vectors in grouped.items()
        if vectors
    }


def _evaluate_accuracy(
    core: BioARNCore,
    reference_samples: list[tuple[torch.Tensor, int]],
    eval_samples: list[tuple[torch.Tensor, int]],
) -> tuple[float, float]:
    prototypes = _build_label_prototypes(core, reference_samples)
    correct = 0
    abstained = 0
    for vector, label in eval_samples:
        recognition = core.recognize(vector)
        if recognition.abstained or not prototypes:
            abstained += 1
            continue
        labels = list(prototypes.keys())
        similarities = torch.tensor(
            [
                float(
                    cosine_similarity(
                        prototypes[current_label].unsqueeze(0),
                        recognition.concept_direction.unsqueeze(0),
                    ).item()
                )
                for current_label in labels
            ],
            dtype=torch.float32,
        )
        predicted = labels[int(torch.argmax(similarities).item())]
        correct += int(predicted == label)
    total = len(eval_samples)
    return correct / max(total, 1), abstained / max(total, 1)


class TestFullSystemE2E:
    """End-to-end tests of the complete Bio-ARN system."""

    def test_mnist_end_to_end(self, small_config, sample_mnist_data) -> None:
        core = BioARNCore(deepcopy(small_config))
        trainer = OnlineTrainer(log_every=1_000, checkpoint_every=1_000)

        train_result = trainer.train(core, sample_mnist_data.train_stream, deepcopy(small_config))
        eval_result = trainer.evaluate(
            core,
            sample_mnist_data.eval_stream,
            label_prototypes=train_result.metrics["label_prototypes"],
        )

        assert train_result.total_steps == 100
        assert eval_result.accuracy > 0.50

    def test_perception_to_generation(self, small_config, sample_mnist_data) -> None:
        loop = SensorimotorLoop(deepcopy(small_config))
        visual_input = sample_mnist_data.visual_seed

        sensory = loop.sense(visual_input=visual_input)
        _ = loop.predict(sensory.features)
        recognition = loop.recognize(sensory.features)
        assert loop._last_perception is not None  # noqa: SLF001
        attention = loop.attend(loop._last_perception)  # noqa: SLF001
        plan = loop.plan(recognition.concept_direction)
        action = loop.act(plan)
        generated = loop.generate_text(recognition.concept_direction, max_tokens=12)

        assert sensory.visual_output is not None
        assert attention.broadcast.num_occupied >= 1
        assert action.generated is not None
        assert generated.strip()

    def test_online_learning_loop(self, small_config, sample_mnist_data) -> None:
        core = BioARNCore(deepcopy(small_config))

        for vector, _label in sample_mnist_data.early_stream:
            core.forward(vector, learn=True)
        accuracy_before, _ = _evaluate_accuracy(
            core,
            sample_mnist_data.early_stream,
            sample_mnist_data.eval_stream,
        )

        for vector, _label in sample_mnist_data.late_stream:
            core.forward(vector, learn=True)
        accuracy_after, _ = _evaluate_accuracy(
            core,
            sample_mnist_data.train_stream,
            sample_mnist_data.eval_stream,
        )

        assert accuracy_after > accuracy_before
        assert accuracy_after >= 0.50

    def test_abstention_on_noise(self, trained_system) -> None:
        abstentions = 0
        for noise in trained_system.data.noise_stream:
            abstentions += int(trained_system.system.recognize(noise).abstained)
        abstention_rate = abstentions / len(trained_system.data.noise_stream)

        assert abstention_rate >= 0.70

    def test_one_shot_novel_concept(self, small_config, sample_mnist_data) -> None:
        core = BioARNCore(deepcopy(small_config))
        for vector, _label in sample_mnist_data.train_stream[:20]:
            core.forward(vector, learn=True)

        before = core.recognize(sample_mnist_data.novel_pattern)
        core.forward(sample_mnist_data.novel_pattern, learn=True)
        after = core.recognize(sample_mnist_data.novel_pattern)

        assert before.abstained is True
        assert after.abstained is False
        assert after.confidence > 0.0
        assert torch.count_nonzero(after.concept_direction).item() > 0

    def test_continual_no_forget(self, small_config, sample_mnist_data) -> None:
        evaluator = ContinualLearningEvaluator(BioARNCore(deepcopy(small_config)))

        result = evaluator.run_sequential_test(
            [
                [sample_mnist_data.class_a_batch],
                [sample_mnist_data.class_b_batch],
            ]
        )

        assert result.mean_forgetting <= 0.05
        assert result.stage_accuracies[-1][0] >= 0.95
        assert result.stage_accuracies[-1][1] >= 0.95

    def test_sensorimotor_loop_runs(self, small_config, sample_mnist_data) -> None:
        loop = SensorimotorLoop(deepcopy(small_config))
        outputs = []
        for step in range(10):
            visual = sample_mnist_data.visual_seed if step % 2 == 0 else sample_mnist_data.reward_sequence[step % len(sample_mnist_data.reward_sequence)]
            outputs.append(loop.step(visual_input=visual))

        assert len(outputs) == 10
        assert outputs[-1].timestep == 9

    def test_generation_produces_output(self, small_config, sample_mnist_data) -> None:
        loop = SensorimotorLoop(deepcopy(small_config))
        seed_step = loop.step(visual_input=sample_mnist_data.visual_seed)

        text = loop.generate_text(seed_step.recognition.concept_direction, max_tokens=12)

        assert isinstance(text, str)
        assert text.strip()

    def test_checkpoint_resume(self, tmp_path, small_config, sample_mnist_data) -> None:
        manager = CheckpointManager()
        resumed_core = BioARNCore(deepcopy(small_config))
        for vector, _label in sample_mnist_data.train_stream[:50]:
            resumed_core.forward(vector, learn=True)

        checkpoint_path = tmp_path / "bioarn-core.pt"
        manager.save(resumed_core, checkpoint_path)
        loaded_core = manager.load(checkpoint_path)
        for vector, _label in sample_mnist_data.train_stream[50:]:
            loaded_core.forward(vector, learn=True)
        resumed_accuracy, _ = _evaluate_accuracy(
            loaded_core,
            sample_mnist_data.train_stream,
            sample_mnist_data.eval_stream,
        )

        reference_core = BioARNCore(deepcopy(small_config))
        for vector, _label in sample_mnist_data.train_stream:
            reference_core.forward(vector, learn=True)
        reference_accuracy, _ = _evaluate_accuracy(
            reference_core,
            sample_mnist_data.train_stream,
            sample_mnist_data.eval_stream,
        )

        assert resumed_accuracy + 0.05 >= reference_accuracy

    def test_scaled_system_equivalence(self, small_config, sample_mnist_data) -> None:
        original = BioARNCore(deepcopy(small_config))
        scaled = ScaledBioARN(deepcopy(small_config), use_optimized=True)
        sample = sample_mnist_data.eval_stream[0][0]

        original_perception = original.perceive(sample)
        scaled_perception = scaled.perceive(sample)

        assert original_perception.num_fired == scaled_perception.num_fired
        assert original_perception.pool_output.recruited_index == scaled_perception.pool_output.recruited_index
        assert torch.allclose(
            original_perception.vote_result.winning_direction,
            scaled_perception.vote_result.winning_direction,
            atol=1e-5,
        )

    def test_reward_modulates_learning(self, small_config, sample_mnist_data) -> None:
        baseline_loop = SensorimotorLoop(deepcopy(small_config))
        rewarded_loop = SensorimotorLoop(deepcopy(small_config))
        pattern = sample_mnist_data.visual_seed

        baseline_loop.step(visual_input=pattern)
        rewarded_loop.step(visual_input=pattern)

        baseline_before = baseline_loop.core.ccc_pool.cccs[0].feedback_weights.detach().clone()
        rewarded_before = rewarded_loop.core.ccc_pool.cccs[0].feedback_weights.detach().clone()

        baseline_loop.step(visual_input=pattern)
        rewarded_loop.reward.apply_external_reward(1.0)
        rewarded_step = rewarded_loop.step(visual_input=pattern)

        baseline_delta = (
            baseline_loop.core.ccc_pool.cccs[0].feedback_weights.detach() - baseline_before
        ).norm().item()
        rewarded_delta = (
            rewarded_loop.core.ccc_pool.cccs[0].feedback_weights.detach() - rewarded_before
        ).norm().item()

        assert rewarded_delta > baseline_delta
        assert rewarded_step.reward.modulation.learning_rate_multiplier > 1.0
        assert rewarded_loop.core.config.ccc.slow_lr > baseline_loop.core.config.ccc.slow_lr

    def test_full_pipeline_no_backprop(self, small_config, sample_mnist_data) -> None:
        loop = SensorimotorLoop(deepcopy(small_config))

        output = loop.step(visual_input=sample_mnist_data.visual_seed)

        assert output.learned is True
        assert all(parameter.grad is None for parameter in loop.parameters())
