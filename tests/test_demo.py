from __future__ import annotations

import torch

from demo.app import create_app
from demo.models import (
    create_live_learning_session,
    get_energy_dashboard_data,
    get_mnist_model,
    get_multimodal_model,
    get_text_model,
)
from demo.visualizations import (
    ccc_activation_map,
    confidence_bar_chart,
    energy_comparison_chart,
    spike_raster_plot,
)


def _zigzag_pattern() -> torch.Tensor:
    image = torch.zeros(28, 28, dtype=torch.float32)
    coords = [(5, 5), (8, 9), (11, 5), (14, 9), (17, 5), (20, 9), (23, 5)]
    for row, col in coords:
        image[row : row + 2, col : col + 4] = 1.0
    return image


def test_mnist_model_loads() -> None:
    model = get_mnist_model()

    assert model.source
    assert int(model.trainer.system.ccc_pool.get_pool_stats()["num_committed"]) > 0


def test_text_model_loads() -> None:
    model = get_text_model()
    result = model.generate_text("The town", max_tokens=12, temperature=0.8, method="greedy")

    assert model.training_tokens > 0
    assert result.generated_text


def test_multimodal_model_loads() -> None:
    model = get_multimodal_model()

    assert len(model.patterns) == 10
    assert model.bindings_learned == 10


def test_digit_recognition() -> None:
    model = get_mnist_model()
    result = model.classify(model.example_digits[3])

    assert result.prediction == 3 or result.abstained
    assert len(result.class_scores) == 10


def test_text_generation() -> None:
    model = get_text_model()
    result = model.generate_text("Rain on", max_tokens=20, temperature=0.9, method="beam")

    assert result.generated_text.strip()
    assert result.tokens_per_sec > 0.0


def test_cross_modal_retrieval() -> None:
    model = get_multimodal_model()
    result = model.retrieve(mode="image-to-text", image=model.patterns["horizontal"])

    assert result.retrieved_text == "horizontal"
    assert result.association_strength > 0.0


def test_live_learning() -> None:
    session = create_live_learning_session()
    outcome = session.teach(_zigzag_pattern(), "zigzag")

    assert "Learned!" in outcome.message
    assert outcome.recognized_label == "zigzag"


def test_no_forgetting_after_live_learn() -> None:
    session = create_live_learning_session()
    session.teach(_zigzag_pattern(), "zigzag")

    assert all(session.retention_report().values())


def test_energy_dashboard_data() -> None:
    dashboard = get_energy_dashboard_data()

    assert dashboard.energies_joules["Bio-ARN (Loihi)"] > 0.0
    assert dashboard.efficiency_callout_x >= 278.0
    assert dashboard.battery_life_hours > 0.0


def test_visualizations_render() -> None:
    assert energy_comparison_chart() is not None
    assert spike_raster_plot(torch.eye(8)) is not None
    assert ccc_activation_map(torch.rand(1, 16)) is not None


def test_confidence_chart() -> None:
    figure = confidence_bar_chart({digit: 0.1 for digit in range(10)})

    assert len(figure.axes[0].patches) == 10


def test_app_launches() -> None:
    app = create_app()

    assert app is not None
