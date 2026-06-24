"""Matplotlib/Plotly charts for the demo."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from demo.models import get_energy_dashboard_data


def confidence_bar_chart(class_scores: dict) -> plt.Figure:
    """Bar chart of class confidence scores."""

    ordered = sorted((int(label), float(score)) for label, score in class_scores.items())
    labels = [str(label) for label, _ in ordered]
    scores = [score for _, score in ordered]
    figure, axis = plt.subplots(figsize=(7, 3.4))
    axis.bar(labels, scores, color="#2563eb")
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Digit class")
    axis.set_ylabel("Confidence")
    axis.set_title("Bio-ARN class confidence")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure


def energy_comparison_chart() -> plt.Figure:
    """Bio-ARN vs GPU vs CPU energy comparison."""

    dashboard = get_energy_dashboard_data()
    labels = list(dashboard.energies_joules.keys())
    values = [dashboard.energies_joules[label] * 1e6 for label in labels]
    colors = ["#10b981", "#f59e0b", "#ef4444"]
    figure, axis = plt.subplots(figsize=(7.2, 4))
    bars = axis.bar(labels, values, color=colors)
    axis.set_ylabel("Energy per inference (µJ)")
    axis.set_title("Projected inference energy")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=False):
        axis.text(
            bar.get_x() + (bar.get_width() / 2),
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.tight_layout()
    return figure


def spike_raster_plot(spikes: Tensor) -> plt.Figure:
    """Visualize spike patterns as a raster plot."""

    tensor = spikes.detach().to(torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)
    spike_indices = torch.nonzero(tensor > 0.1, as_tuple=False)
    figure, axis = plt.subplots(figsize=(7, 3.2))
    if spike_indices.numel() > 0:
        axis.scatter(spike_indices[:, 1].cpu(), spike_indices[:, 0].cpu(), s=6, c="#111827")
    axis.set_xlabel("Neuron / feature index")
    axis.set_ylabel("Timestep")
    axis.set_title("Spike raster")
    axis.invert_yaxis()
    figure.tight_layout()
    return figure


def ccc_activation_map(pool_activations: Tensor) -> plt.Figure:
    """Heatmap of CCC activations."""

    tensor = pool_activations.detach().to(torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)
    figure, axis = plt.subplots(figsize=(7, 2.8))
    heatmap = axis.imshow(tensor.cpu().numpy(), cmap="viridis", aspect="auto")
    axis.set_xlabel("CCC index")
    axis.set_ylabel("Activation row")
    axis.set_title("CCC activation map")
    figure.colorbar(heatmap, ax=axis, fraction=0.04, pad=0.04)
    figure.tight_layout()
    return figure


__all__ = [
    "ccc_activation_map",
    "confidence_bar_chart",
    "energy_comparison_chart",
    "spike_raster_plot",
]
