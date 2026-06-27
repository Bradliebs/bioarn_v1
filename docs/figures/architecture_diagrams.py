from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


OUTPUT_DIR = Path(__file__).resolve().parent

COLORS = {
    "blue": "#60A5FA",
    "blue_dark": "#1D4ED8",
    "green": "#86EFAC",
    "green_dark": "#15803D",
    "orange": "#FDBA74",
    "orange_dark": "#C2410C",
    "purple": "#C4B5FD",
    "purple_dark": "#6D28D9",
    "red": "#FCA5A5",
    "red_dark": "#B91C1C",
    "gray": "#E2E8F0",
    "gray_dark": "#475569",
    "text": "#0F172A",
    "muted": "#334155",
    "line": "#64748B",
    "grid": "#CBD5E1",
    "canvas": "#F8FAFC",
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": COLORS["canvas"],
        "savefig.facecolor": "white",
    }
)


def configure_axis(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title(title, loc="left", pad=18)
    if subtitle:
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color=COLORS["muted"],
        )


def add_box(
    ax,
    center: tuple[float, float],
    size: tuple[float, float],
    text: str,
    facecolor: str,
    *,
    edgecolor: str | None = None,
    linewidth: float = 2.0,
    fontsize: int = 11,
    fontweight: str = "bold",
    textcolor: str = COLORS["text"],
    zorder: int = 2,
) -> tuple[float, float, float, float]:
    width, height = size
    x0 = center[0] - (width / 2.0)
    y0 = center[1] - (height / 2.0)
    patch = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.022",
        linewidth=linewidth,
        edgecolor=edgecolor or facecolor,
        facecolor=facecolor,
        mutation_aspect=1.0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        center[0],
        center[1],
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=textcolor,
        zorder=zorder + 1,
    )
    return (x0, y0, width, height)


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    linewidth: float = 2.0,
    linestyle: str = "-",
    connectionstyle: str = "arc3,rad=0.0",
    arrowstyle: str = "-|>",
    mutation_scale: int = 16,
    label: str | None = None,
    label_pos: float = 0.5,
    label_offset: tuple[float, float] = (0.0, 0.0),
    label_color: str | None = None,
    label_size: int = 10,
    zorder: int = 3,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=arrowstyle,
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=4,
        shrinkB=4,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    if label:
        x = start[0] + ((end[0] - start[0]) * label_pos) + label_offset[0]
        y = start[1] + ((end[1] - start[1]) * label_pos) + label_offset[1]
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=label_size,
            color=label_color or color,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.92,
            },
            zorder=zorder + 1,
        )


def add_legend_chip(ax, x: float, y: float, label: str, color: str) -> None:
    add_box(
        ax,
        (x, y),
        (0.15, 0.05),
        label,
        color,
        edgecolor=COLORS["line"],
        linewidth=1.2,
        fontsize=9,
        fontweight="bold",
    )


def add_footer(fig, text: str) -> None:
    fig.text(
        0.01,
        0.01,
        text,
        ha="left",
        va="bottom",
        fontsize=10,
        color=COLORS["muted"],
    )


def save_figure(fig, filename: str) -> Path:
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_figure1_pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(16, 9), constrained_layout=True)
    configure_axis(
        ax,
        "Figure 1. Full Bio-ARN Pipeline",
        "End-to-end flow from sensory encoding through precision-weighted learning and neuromorphic export.",
    )

    input_box = add_box(ax, (0.07, 0.74), (0.10, 0.09), "Input", COLORS["gray"], edgecolor=COLORS["gray_dark"])
    attention_box = add_box(
        ax,
        (0.20, 0.74),
        (0.14, 0.10),
        "Spatial\nAttention",
        COLORS["blue"],
        edgecolor=COLORS["blue_dark"],
    )
    hierarchy_panel = add_box(
        ax,
        (0.43, 0.74),
        (0.31, 0.18),
        "V1 → V2 → V4 → IT Hierarchy",
        "#DBEAFE",
        edgecolor=COLORS["blue_dark"],
        fontsize=13,
    )
    ccc_box = add_box(
        ax,
        (0.66, 0.74),
        (0.13, 0.10),
        "CCC Pool",
        COLORS["orange"],
        edgecolor=COLORS["orange_dark"],
    )
    gnw_box = add_box(
        ax,
        (0.84, 0.74),
        (0.16, 0.10),
        "GNW\nWorkspace",
        COLORS["purple"],
        edgecolor=COLORS["purple_dark"],
    )

    stage_y = 0.70
    stage_width = 0.055
    for idx, (x, label) in enumerate(((0.33, "V1"), (0.40, "V2"), (0.47, "V4"), (0.54, "IT"))):
        add_box(
            ax,
            (x, stage_y),
            (stage_width, 0.06),
            label,
            "#BFDBFE",
            edgecolor=COLORS["blue_dark"],
            linewidth=1.3,
            fontsize=9,
        )
        if idx < 3:
            add_arrow(ax, (x + 0.031, stage_y), (x + 0.068, stage_y), color=COLORS["blue_dark"], linewidth=1.6)
    add_arrow(
        ax,
        (0.545, 0.665),
        (0.325, 0.665),
        color=COLORS["blue_dark"],
        linewidth=1.4,
        linestyle="--",
        connectionstyle="arc3,rad=-0.18",
        label="top-down predictions",
        label_pos=0.45,
        label_offset=(0.0, -0.04),
        label_color=COLORS["blue_dark"],
        label_size=9,
    )

    prediction_box = add_box(
        ax,
        (0.43, 0.45),
        (0.19, 0.10),
        "Prediction Error\nGate",
        COLORS["green"],
        edgecolor=COLORS["green_dark"],
    )
    entropy_box = add_box(
        ax,
        (0.84, 0.48),
        (0.18, 0.10),
        "Pool Entropy\nEstimator",
        "#E9D5FF",
        edgecolor=COLORS["purple_dark"],
    )
    precision_box = add_box(
        ax,
        (0.84, 0.33),
        (0.16, 0.09),
        "Precision\nSignal",
        "#DCFCE7",
        edgecolor=COLORS["green_dark"],
    )
    lr_box = add_box(
        ax,
        (0.68, 0.18),
        (0.21, 0.10),
        "Learning Rate\nModulation",
        COLORS["green"],
        edgecolor=COLORS["green_dark"],
    )
    lock_box = add_box(
        ax,
        (0.88, 0.18),
        (0.15, 0.10),
        "Concept\nLocking",
        "#FED7AA",
        edgecolor=COLORS["orange_dark"],
    )
    ensemble_box = add_box(
        ax,
        (0.69, 0.05),
        (0.17, 0.09),
        "Ensemble /\nOOD",
        COLORS["purple"],
        edgecolor=COLORS["purple_dark"],
    )
    classify_box = add_box(
        ax,
        (0.88, 0.08),
        (0.15, 0.09),
        "Classification",
        COLORS["red"],
        edgecolor=COLORS["red_dark"],
    )
    loihi_box = add_box(
        ax,
        (0.88, 0.005 + 0.045),
        (0.15, 0.08),
        "Loihi 2\nExport",
        "#FECACA",
        edgecolor=COLORS["red_dark"],
    )

    add_arrow(ax, (0.12, 0.74), (0.13, 0.74), color=COLORS["line"], linewidth=2.2)
    add_arrow(ax, (0.27, 0.74), (0.285, 0.74), color=COLORS["blue_dark"], linewidth=2.2)
    add_arrow(ax, (0.585, 0.74), (0.595, 0.74), color=COLORS["blue_dark"], linewidth=2.2)
    add_arrow(ax, (0.725, 0.74), (0.755, 0.74), color=COLORS["orange_dark"], linewidth=2.2)

    add_arrow(
        ax,
        (0.43, 0.64),
        (0.43, 0.51),
        color=COLORS["green_dark"],
        linewidth=2.1,
        label="predictive coding",
        label_pos=0.42,
        label_offset=(-0.11, 0.0),
        label_color=COLORS["green_dark"],
    )
    add_arrow(
        ax,
        (0.47, 0.50),
        (0.47, 0.64),
        color=COLORS["green_dark"],
        linewidth=1.8,
        linestyle="--",
        connectionstyle="arc3,rad=0.05",
    )

    add_arrow(
        ax,
        (0.84, 0.68),
        (0.84, 0.54),
        color=COLORS["purple_dark"],
        linewidth=2.1,
        label="precision weighting",
        label_pos=0.45,
        label_offset=(0.11, 0.0),
        label_color=COLORS["purple_dark"],
    )
    add_arrow(
        ax,
        (0.88, 0.54),
        (0.88, 0.68),
        color=COLORS["purple_dark"],
        linewidth=1.8,
        linestyle="--",
        connectionstyle="arc3,rad=-0.05",
    )

    add_arrow(ax, (0.84, 0.43), (0.84, 0.38), color=COLORS["green_dark"], linewidth=2.1)
    add_arrow(ax, (0.80, 0.29), (0.74, 0.22), color=COLORS["green_dark"], linewidth=2.1)
    add_arrow(ax, (0.76, 0.18), (0.80, 0.18), color=COLORS["orange_dark"], linewidth=2.1)
    add_arrow(ax, (0.68, 0.13), (0.69, 0.10), color=COLORS["green_dark"], linewidth=2.0)
    add_arrow(ax, (0.78, 0.05), (0.80, 0.08), color=COLORS["purple_dark"], linewidth=2.0)
    add_arrow(ax, (0.88, 0.04), (0.88, 0.02), color=COLORS["red_dark"], linewidth=2.0)

    add_arrow(
        ax,
        (0.89, 0.13),
        (0.67, 0.69),
        color=COLORS["orange_dark"],
        linewidth=1.5,
        linestyle=":",
        connectionstyle="arc3,rad=0.18",
        label="stabilized concepts",
        label_pos=0.56,
        label_offset=(0.02, -0.03),
        label_color=COLORS["orange_dark"],
        label_size=9,
    )

    ax.text(
        0.67,
        0.89,
        "Processing bands",
        fontsize=10,
        fontweight="bold",
        color=COLORS["muted"],
    )
    add_legend_chip(ax, 0.67, 0.84, "Sensory", COLORS["blue"])
    add_legend_chip(ax, 0.84, 0.84, "Learning", COLORS["green"])
    add_legend_chip(ax, 0.67, 0.78, "Memory", COLORS["orange"])
    add_legend_chip(ax, 0.84, 0.78, "Global", COLORS["purple"])
    add_legend_chip(ax, 0.76, 0.72, "Outputs", COLORS["red"])

    ax.text(
        0.03,
        0.17,
        "Sparse sensory abstraction\nfeeds CCC competition.\nWorkspace entropy then\nscales local plasticity\nbefore classification\nand Loihi deployment.",
        fontsize=10,
        color=COLORS["muted"],
        ha="left",
        va="bottom",
        linespacing=1.4,
    )

    add_footer(
        fig,
        "Blue = sensory hierarchy | Green = precision-weighted learning | Orange = memory management | Purple = global integration | Red = outputs",
    )
    return save_figure(fig, "figure1_full_bioarn_pipeline.png")


def create_figure2_ccc() -> Path:
    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    configure_axis(
        ax,
        "Figure 2. CCC Internal Architecture",
        "Single-concept processing path showing abstention, recruitment, resonance, and slow Hebbian refinement.",
    )

    add_box(ax, (0.09, 0.74), (0.12, 0.09), "Raw\nInput", COLORS["gray"], edgecolor=COLORS["gray_dark"])
    add_box(ax, (0.27, 0.74), (0.18, 0.10), "F1 Sparse\nEncoder", COLORS["blue"], edgecolor=COLORS["blue_dark"])
    add_box(ax, (0.47, 0.74), (0.18, 0.10), "F2 Concept\nSpace", "#DBEAFE", edgecolor=COLORS["blue_dark"])
    add_box(
        ax,
        (0.68, 0.74),
        (0.18, 0.10),
        "Margin Gate",
        COLORS["orange"],
        edgecolor=COLORS["orange_dark"],
    )
    add_box(ax, (0.83, 0.74), (0.12, 0.08), "Fire?", "#FFEDD5", edgecolor=COLORS["orange_dark"])

    add_arrow(ax, (0.15, 0.74), (0.18, 0.74), color=COLORS["line"], linewidth=2.1)
    add_arrow(ax, (0.36, 0.74), (0.38, 0.74), color=COLORS["blue_dark"], linewidth=2.1)
    add_arrow(ax, (0.56, 0.74), (0.59, 0.74), color=COLORS["blue_dark"], linewidth=2.1)
    add_arrow(ax, (0.77, 0.74), (0.78, 0.74), color=COLORS["orange_dark"], linewidth=2.1)

    add_box(
        ax,
        (0.83, 0.53),
        (0.18, 0.10),
        "Feedback\nPrediction",
        COLORS["purple"],
        edgecolor=COLORS["purple_dark"],
    )
    add_box(
        ax,
        (0.83, 0.34),
        (0.18, 0.10),
        "Resonance\nCheck",
        "#E9D5FF",
        edgecolor=COLORS["purple_dark"],
    )
    add_box(
        ax,
        (0.62, 0.15),
        (0.21, 0.10),
        "learn_slow()\nHebbian refine",
        COLORS["green"],
        edgecolor=COLORS["green_dark"],
    )
    add_box(
        ax,
        (0.84, 0.15),
        (0.18, 0.10),
        "Concept Locking\nCheck",
        "#DCFCE7",
        edgecolor=COLORS["green_dark"],
    )
    add_box(
        ax,
        (0.40, 0.47),
        (0.22, 0.11),
        "Recruit new CCC\nlearn_fast()",
        "#FED7AA",
        edgecolor=COLORS["orange_dark"],
    )

    add_arrow(
        ax,
        (0.83, 0.70),
        (0.83, 0.58),
        color=COLORS["purple_dark"],
        linewidth=2.1,
        label="Yes",
        label_pos=0.2,
        label_offset=(0.05, 0.0),
        label_color=COLORS["purple_dark"],
    )
    add_arrow(
        ax,
        (0.78, 0.70),
        (0.46, 0.52),
        color=COLORS["orange_dark"],
        linewidth=2.0,
        connectionstyle="arc3,rad=0.16",
        label="No",
        label_pos=0.45,
        label_offset=(-0.02, 0.04),
        label_color=COLORS["orange_dark"],
    )
    add_arrow(ax, (0.83, 0.48), (0.83, 0.39), color=COLORS["purple_dark"], linewidth=2.1)
    add_arrow(ax, (0.78, 0.29), (0.70, 0.20), color=COLORS["green_dark"], linewidth=2.1)
    add_arrow(ax, (0.73, 0.15), (0.75, 0.15), color=COLORS["green_dark"], linewidth=2.1)

    add_arrow(
        ax,
        (0.83, 0.29),
        (0.30, 0.69),
        color=COLORS["purple_dark"],
        linewidth=1.6,
        linestyle="--",
        connectionstyle="arc3,rad=0.35",
        label="feedback closes the resonance loop",
        label_pos=0.45,
        label_offset=(-0.04, 0.03),
        label_color=COLORS["purple_dark"],
        label_size=9,
    )

    ax.text(
        0.08,
        0.58,
        "Architectural abstention:\nno match means no forced label.",
        ha="left",
        va="center",
        fontsize=11,
        color=COLORS["muted"],
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": COLORS["grid"]},
    )
    ax.text(
        0.08,
        0.36,
        "Gate criterion\ncos(h, d) > θ_margin",
        ha="left",
        va="center",
        fontsize=11,
        color=COLORS["text"],
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#FFF7ED", "edgecolor": COLORS["orange"]},
    )
    ax.text(
        0.08,
        0.16,
        "Only resonant committed concepts\nreceive slow local updates.\nUnmatched samples recruit capacity.",
        ha="left",
        va="center",
        fontsize=10,
        color=COLORS["muted"],
        linespacing=1.4,
    )
    ax.text(
        0.84,
        0.05,
        "importance > lock_threshold ?",
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["green_dark"],
    )

    add_footer(fig, "CCCs combine sparse encoding, explicit abstention, top-down resonance, one-shot recruitment, and local Hebbian refinement.")
    return save_figure(fig, "figure2_ccc_internal_architecture.png")


def create_figure3_precision() -> Path:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 7),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.3, 1.0]},
    )
    ax, panel = axes
    configure_axis(
        ax,
        "Figure 3. Precision-Weighted Predictive Processing",
        "Frank et al.-inspired uncertainty signal that scales CCC plasticity from recent winner entropy.",
    )
    configure_axis(panel, "", None)

    add_box(
        ax,
        (0.12, 0.72),
        (0.23, 0.11),
        "CCC Pool\nFiring History",
        COLORS["orange"],
        edgecolor=COLORS["orange_dark"],
    )
    add_box(
        ax,
        (0.39, 0.72),
        (0.20, 0.11),
        "Entropy\nEstimator",
        COLORS["purple"],
        edgecolor=COLORS["purple_dark"],
    )
    add_box(
        ax,
        (0.66, 0.72),
        (0.20, 0.11),
        "Pool Entropy H",
        "#E9D5FF",
        edgecolor=COLORS["purple_dark"],
    )
    add_box(
        ax,
        (0.66, 0.49),
        (0.25, 0.11),
        "Precision Signal\nσ(α(H-θ))",
        COLORS["green"],
        edgecolor=COLORS["green_dark"],
    )
    add_box(
        ax,
        (0.18, 0.25),
        (0.22, 0.11),
        "Prediction Error\nδ",
        "#DBEAFE",
        edgecolor=COLORS["blue_dark"],
    )
    add_box(
        ax,
        (0.64, 0.25),
        (0.30, 0.12),
        "Weighted Learning Signal\nδ × precision",
        "#DCFCE7",
        edgecolor=COLORS["green_dark"],
    )

    add_arrow(ax, (0.24, 0.72), (0.28, 0.72), color=COLORS["purple_dark"], linewidth=2.1)
    add_arrow(ax, (0.49, 0.72), (0.55, 0.72), color=COLORS["purple_dark"], linewidth=2.1)
    add_arrow(ax, (0.66, 0.66), (0.66, 0.55), color=COLORS["green_dark"], linewidth=2.1)
    add_arrow(ax, (0.29, 0.25), (0.48, 0.25), color=COLORS["blue_dark"], linewidth=2.1, label="×", label_pos=0.5, label_offset=(0.0, 0.06), label_color=COLORS["text"], label_size=14)
    add_arrow(ax, (0.61, 0.44), (0.57, 0.30), color=COLORS["green_dark"], linewidth=2.1, connectionstyle="arc3,rad=0.12")
    add_arrow(ax, (0.46, 0.25), (0.49, 0.25), color=COLORS["green_dark"], linewidth=2.1)

    ax.text(
        0.08,
        0.07,
        "High H → fast learning for novel winner patterns\nLow H → protect memory for familiar winner patterns",
        ha="left",
        va="bottom",
        fontsize=11,
        color=COLORS["muted"],
        linespacing=1.5,
    )

    inset = inset_axes(ax, width="37%", height="34%", loc="upper right", borderpad=1.2)
    inset.set_xlim(0.0, 1.0)
    inset.set_ylim(0.0, 1.0)
    inset.axis("off")
    inset.set_facecolor("white")
    add_box(inset, (0.27, 0.62), (0.34, 0.24), "Frank et al.\nhippocampal\nripples", "#FEE2E2", edgecolor=COLORS["red_dark"], linewidth=1.2, fontsize=9)
    add_box(inset, (0.74, 0.62), (0.34, 0.24), "Bio-ARN\npool entropy", "#E9D5FF", edgecolor=COLORS["purple_dark"], linewidth=1.2, fontsize=9)
    add_arrow(inset, (0.44, 0.62), (0.57, 0.62), color=COLORS["muted"], linewidth=1.6, label="analogy", label_pos=0.5, label_offset=(0.0, 0.10), label_color=COLORS["muted"], label_size=8)
    inset.text(0.5, 0.22, "Both act as precision signals\nthat scale learning from surprise.", ha="center", va="center", fontsize=8.5, color=COLORS["muted"])

    panel.text(
        0.02,
        0.88,
        "Mechanism",
        fontsize=13,
        fontweight="bold",
        color=COLORS["text"],
    )
    panel.text(
        0.02,
        0.72,
        "1. Track which CCCs win over a\n   recent temporal window.\n\n2. Compute normalized entropy H.\n\n3. Convert H to a smooth precision\n   signal with a sigmoid.\n\n4. Multiply local prediction-error\n   learning by the current precision.",
        fontsize=11,
        color=COLORS["muted"],
        va="top",
        linespacing=1.55,
    )
    add_box(panel, (0.50, 0.25), (0.86, 0.24), "Novel state:\nspread-out winners\n→ high entropy\n→ high precision\n→ larger updates", "#ECFCCB", edgecolor=COLORS["green_dark"], fontsize=10, fontweight="bold")
    add_box(panel, (0.50, 0.08), (0.86, 0.14), "Familiar state:\nrepeat winners → low entropy → protect memory", "#FEF3C7", edgecolor=COLORS["orange_dark"], fontsize=10, fontweight="bold")

    add_footer(fig, "Precision is derived from recent CCC winner entropy rather than a global backward pass, keeping the mechanism local and online.")
    return save_figure(fig, "figure3_precision_weighted_predictive_processing.png")


def create_figure4_locking() -> Path:
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
    configure_axis(
        ax,
        "Figure 4. Concept Locking Lifecycle",
        "CCC state transition from free slot to permanent memory cell once importance crosses the lock threshold.",
    )

    add_box(ax, (0.10, 0.62), (0.15, 0.11), "Uncommitted", COLORS["gray"], edgecolor=COLORS["gray_dark"])
    add_box(ax, (0.33, 0.62), (0.22, 0.11), "Recruit\nlearn_fast()", COLORS["orange"], edgecolor=COLORS["orange_dark"])
    add_box(ax, (0.55, 0.62), (0.15, 0.11), "Committed", "#DBEAFE", edgecolor=COLORS["blue_dark"])
    add_box(ax, (0.77, 0.62), (0.24, 0.11), "Hebbian updates\nlearn_slow()", COLORS["green"], edgecolor=COLORS["green_dark"])
    add_box(ax, (0.93, 0.62), (0.12, 0.11), "LOCKED", "#FECACA", edgecolor=COLORS["red_dark"])

    add_arrow(ax, (0.18, 0.62), (0.22, 0.62), color=COLORS["orange_dark"], linewidth=2.2)
    add_arrow(ax, (0.44, 0.62), (0.47, 0.62), color=COLORS["blue_dark"], linewidth=2.2)
    add_arrow(ax, (0.63, 0.62), (0.65, 0.62), color=COLORS["green_dark"], linewidth=2.2)
    add_arrow(ax, (0.89, 0.62), (0.87, 0.62), color=COLORS["red_dark"], linewidth=2.2, connectionstyle="arc3,rad=0.0")

    add_box(ax, (0.55, 0.34), (0.33, 0.11), "importance grows with fires + confidence + recency", "#E0F2FE", edgecolor=COLORS["blue_dark"], fontsize=11)
    add_box(ax, (0.84, 0.34), (0.24, 0.11), "importance >\nlock_threshold", "#FEF3C7", edgecolor=COLORS["orange_dark"], fontsize=11)
    add_box(ax, (0.93, 0.15), (0.16, 0.12), "Still fires\nNever updates", "#FEE2E2", edgecolor=COLORS["red_dark"], fontsize=11)

    add_arrow(ax, (0.55, 0.56), (0.55, 0.40), color=COLORS["blue_dark"], linewidth=2.0)
    add_arrow(ax, (0.72, 0.34), (0.75, 0.34), color=COLORS["orange_dark"], linewidth=2.0)
    add_arrow(ax, (0.90, 0.39), (0.92, 0.56), color=COLORS["red_dark"], linewidth=2.0, connectionstyle="arc3,rad=0.15")
    add_arrow(ax, (0.93, 0.56), (0.93, 0.22), color=COLORS["red_dark"], linewidth=2.0, linestyle="--")

    ax.text(
        0.08,
        0.16,
        "Locked CCCs remain active detectors during inference\nbut skip both learn_fast() and learn_slow() updates.",
        fontsize=11,
        color=COLORS["muted"],
        ha="left",
        va="center",
        linespacing=1.5,
    )

    add_footer(fig, "Concept locking operationalizes stability-plasticity control: mature CCCs remain readable while their weights become immutable.")
    return save_figure(fig, "figure4_concept_locking_lifecycle.png")


def create_figure5_energy() -> Path:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )
    ax, panel = axes

    values = [179.65e-6, 50.01e-3]
    labels = ["Bio-ARN\non Loihi 2", "Transformer\non A100 GPU"]
    colors = [COLORS["blue_dark"], COLORS["red_dark"]]

    bars = ax.bar(labels, values, color=colors, width=0.58, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("Energy per inference (J, log scale)")
    ax.set_title("Figure 5. Energy Comparison", loc="left", pad=14)
    ax.text(
        0.0,
        1.02,
        "Projected inference-energy comparison at the matched benchmark tier.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=COLORS["muted"],
    )
    ax.grid(axis="y", which="both", linestyle="--", linewidth=0.8, color=COLORS["grid"], zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])

    for bar, value, label in zip(bars, values, ("179.65 µJ", "50.01 mJ"), strict=True):
        ax.text(
            bar.get_x() + (bar.get_width() / 2.0),
            value * 1.25,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=COLORS["text"],
        )

    ax.annotate(
        "278× lower\nprojected energy",
        xy=(0, values[0] * 1.8),
        xytext=(0.50, values[1] / 3.5),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "color": COLORS["muted"], "linewidth": 1.8},
        fontsize=11,
        fontweight="bold",
        color=COLORS["muted"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": COLORS["grid"]},
    )

    configure_axis(panel, "", None)
    panel.text(0.02, 0.88, "Takeaway", fontsize=13, fontweight="bold", color=COLORS["text"])
    add_box(panel, (0.50, 0.64), (0.88, 0.22), "Bio-ARN on Loihi 2:\n179.65 µJ / inference", "#DBEAFE", edgecolor=COLORS["blue_dark"], fontsize=12)
    add_box(panel, (0.50, 0.39), (0.88, 0.22), "Transformer on A100:\n50.01 mJ / inference", "#FEE2E2", edgecolor=COLORS["red_dark"], fontsize=12)
    add_box(panel, (0.50, 0.14), (0.88, 0.20), "Ratio ≈ 278× energy advantage\nfor the projected neuromorphic path.", "#F8FAFC", edgecolor=COLORS["grid"], fontsize=11)

    add_footer(fig, "Bio-ARN: 179.65 µJ/inference on projected Loihi 2 vs 50.01 mJ/inference for a matched transformer on A100.")
    return save_figure(fig, "figure5_energy_comparison.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = [
        create_figure1_pipeline(),
        create_figure2_ccc(),
        create_figure3_precision(),
        create_figure4_locking(),
        create_figure5_energy(),
    ]
    print("Generated architecture figures:")
    for path in generated:
        print(f" - {path}")


if __name__ == "__main__":
    main()
