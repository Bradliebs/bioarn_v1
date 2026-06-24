from __future__ import annotations

import torch

from bioarn.ensemble import DiversityManager, EnsembleConfig, EnsemblePool, HebbianBoosting
from bioarn.training import SyntheticCIFAR10Stream, take_samples


class MockExpert:
    def __init__(self, classify_fn):
        self.classify_fn = classify_fn
        self.learn_calls: list[tuple[torch.Tensor, int | None]] = []

    def classify(self, image: torch.Tensor):
        return self.classify_fn(image)

    def learn(self, image: torch.Tensor, label: int | None = None) -> None:
        self.learn_calls.append((image.detach().clone(), label))


class LookupExpert:
    def __init__(self, predictions: list[int], confidence: float = 0.8):
        self.predictions = list(predictions)
        self.confidence = float(confidence)

    def classify(self, image: torch.Tensor):
        index = int(image.reshape(-1)[0].item())
        return self.predictions[index], self.confidence

    def learn(self, image: torch.Tensor, label: int | None = None) -> None:
        del image, label


def test_ensemble_init() -> None:
    ensemble = EnsemblePool(EnsembleConfig(num_experts=3))

    assert ensemble.config.num_experts == 3
    assert ensemble.experts == []
    assert ensemble.get_expert_accuracies() == {}


def test_add_expert() -> None:
    ensemble = EnsemblePool(EnsembleConfig(num_experts=2))
    ensemble.add_expert("a", MockExpert(lambda _: (0, 0.8)))
    ensemble.add_expert("b", MockExpert(lambda _: (1, 0.7)))

    assert [expert.name for expert in ensemble.experts] == ["a", "b"]


def test_ensemble_classify() -> None:
    ensemble = EnsemblePool(EnsembleConfig(num_experts=3, voting_method="weighted"))
    ensemble.add_expert("a", MockExpert(lambda _: (1, 0.9)))
    ensemble.add_expert("b", MockExpert(lambda _: (1, 0.6)))
    ensemble.add_expert("c", MockExpert(lambda _: (-1, 0.0, True)))

    result = ensemble.classify(torch.tensor([1.0, 0.0]))

    assert result.predicted_class == 1
    assert result.abstained is False
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.expert_results) == 3


def test_ensemble_abstains() -> None:
    ensemble = EnsemblePool(EnsembleConfig(num_experts=3))
    ensemble.add_expert("a", MockExpert(lambda _: (-1, 0.0, True)))
    ensemble.add_expert("b", MockExpert(lambda _: (-1, 0.0, True)))
    ensemble.add_expert("c", MockExpert(lambda _: (0, 0.9)))

    result = ensemble.classify(torch.tensor([0.0]))

    assert result.abstained is True
    assert result.predicted_class == -1


def test_ensemble_beats_single() -> None:
    labels = [0, 1, 0, 1, 0, 1]
    experts = {
        "a": [0, 1, 0, 1, 1, 0],
        "b": [1, 1, 0, 0, 0, 1],
        "c": [0, 0, 1, 1, 0, 1],
    }
    ensemble = EnsemblePool(EnsembleConfig(num_experts=3, voting_method="majority"))
    for name, predictions in experts.items():
        ensemble.add_expert(name, LookupExpert(predictions))

    ensemble_correct = 0
    single_correct = {name: 0 for name in experts}
    for index, label in enumerate(labels):
        sample = torch.tensor([float(index)])
        result = ensemble.classify(sample)
        ensemble_correct += int(result.predicted_class == label)
        for name, predictions in experts.items():
            single_correct[name] += int(predictions[index] == label)

    assert ensemble_correct > max(single_correct.values())


def test_weighted_voting() -> None:
    ensemble = EnsemblePool(EnsembleConfig(num_experts=2, voting_method="weighted"))
    ensemble.add_expert("low", MockExpert(lambda _: (0, 0.2)))
    ensemble.add_expert("high", MockExpert(lambda _: (1, 0.95)))

    result = ensemble.classify(torch.tensor([0.0]))

    assert result.predicted_class == 1


def _weighted_prediction(boosting: HebbianBoosting, predictions: list[int]) -> int:
    votes: dict[int, float] = {}
    for expert_index, label in enumerate(predictions):
        votes[label] = votes.get(label, 0.0) + float(boosting.get_weights(expert_index)[label].item())
    return max(votes, key=votes.get)


def test_boosting_improves() -> None:
    boosting = HebbianBoosting(num_experts=2, num_classes=2, learning_rate=0.3, penalty_rate=0.12)
    samples = [([0, 1], 1)] * 9 + [([0, 1], 0)] * 3

    baseline = sum(int(_weighted_prediction(HebbianBoosting(2, 2), predictions) == label) for predictions, label in samples)
    for predictions, label in samples:
        boosting.update_weights(predictions, label)
    improved = sum(int(_weighted_prediction(boosting, predictions) == label) for predictions, label in samples)

    assert improved > baseline
    assert boosting.get_weights(0)[0] > boosting.get_weights(0)[1]
    assert boosting.get_weights(1)[1] > boosting.get_weights(1)[0]


def test_diversity_measurement() -> None:
    manager = DiversityManager()
    diversity = manager.measure_diversity([[0, 0, 1], [0, 1, 1], [1, 1, 1]])

    assert 0.0 <= diversity <= 1.0


def test_diverse_experts_disagree() -> None:
    manager = DiversityManager()
    expert_configs = manager.create_diverse_experts(
        {"input_dim": 3072, "concept_dim": 24, "max_pool_size": 18, "image_size": (32, 32, 3), "num_classes": 10},
        num_experts=4,
    )
    ensemble = EnsemblePool(EnsembleConfig(num_experts=4, expert_configs=expert_configs))
    train_samples = take_samples(SyntheticCIFAR10Stream(40, seed=21), 40)
    test_samples = take_samples(SyntheticCIFAR10Stream(12, seed=22, shuffle=False), 12)

    for image, label in train_samples:
        ensemble.learn(image, label)

    prediction_histories = [[] for _ in ensemble.experts]
    for image, _ in test_samples:
        result = ensemble.classify(image)
        for expert_index, expert_result in enumerate(result.expert_results):
            prediction_histories[expert_index].append(expert_result.predicted_class)

    assert manager.measure_diversity(prediction_histories) > 0.0


def test_agreement_correlates_accuracy() -> None:
    labels = [0, 1, 0, 1]
    ensemble = EnsemblePool(EnsembleConfig(num_experts=3, voting_method="majority"))
    ensemble.add_expert("a", LookupExpert([0, 1, 0, 1], confidence=0.9))
    ensemble.add_expert("b", LookupExpert([0, 1, 1, 1], confidence=0.8))
    ensemble.add_expert("c", LookupExpert([0, 1, 1, 0], confidence=0.7))

    high_agreement: list[int] = []
    low_agreement: list[int] = []
    for index, label in enumerate(labels):
        result = ensemble.classify(torch.tensor([float(index)]))
        bucket = high_agreement if result.agreement == 1.0 else low_agreement
        bucket.append(int(result.predicted_class == label))

    assert sum(high_agreement) / len(high_agreement) >= sum(low_agreement) / len(low_agreement)


def test_ensemble_learns() -> None:
    first = MockExpert(lambda _: (0, 0.8))
    second = MockExpert(lambda _: (1, 0.6))
    ensemble = EnsemblePool(EnsembleConfig(num_experts=2))
    ensemble.add_expert("first", first)
    ensemble.add_expert("second", second)
    image = torch.tensor([1.0, 2.0, 3.0])

    ensemble.learn(image, label=4)

    assert len(first.learn_calls) == 1
    assert len(second.learn_calls) == 1
    assert torch.equal(first.learn_calls[0][0], image)
    assert torch.equal(second.learn_calls[0][0], image)
    assert first.learn_calls[0][1] == 4
    assert second.learn_calls[0][1] == 4


def test_no_backprop_ensemble() -> None:
    manager = DiversityManager()
    expert_configs = manager.create_diverse_experts(
        {"input_dim": 3072, "concept_dim": 24, "max_pool_size": 16, "image_size": (32, 32, 3), "num_classes": 10},
        num_experts=2,
    )
    ensemble = EnsemblePool(EnsembleConfig(num_experts=2, expert_configs=expert_configs))
    image, label = take_samples(SyntheticCIFAR10Stream(1, seed=30), 1)[0]

    ensemble.learn(image, label)

    parameters: list[torch.nn.Parameter] = []
    for expert in ensemble.experts:
        pool = expert.pool
        if isinstance(pool, torch.nn.Module):
            parameters.extend(list(pool.parameters()))
        elif hasattr(pool, "layers"):
            for layer in pool.layers:
                parameters.extend(list(layer.pool.core.parameters()))

    assert all(parameter.grad is None for parameter in parameters)
