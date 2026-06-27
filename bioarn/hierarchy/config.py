"""Configuration for hierarchical visual feature learning."""

from __future__ import annotations

from dataclasses import dataclass, field

from bioarn.config import PredictiveConfig, STDPConfig


@dataclass
class HierarchyConfig:
    """Configuration for the ventral-stream-style visual hierarchy."""

    image_size: tuple[int, int, int] = (32, 32, 3)
    num_layers: int = 4

    patch_sizes: list[int] = field(default_factory=lambda: [8, 2, 1, 1])
    pool_sizes: list[int] = field(default_factory=lambda: [100, 200, 500, 200])
    concept_dims: list[int] = field(default_factory=lambda: [32, 64, 128, 64])
    thresholds: list[float] = field(default_factory=lambda: [0.25, 0.3, 0.35, 0.4])
    learning_rates: list[float] = field(default_factory=lambda: [0.05, 0.03, 0.02, 0.01])

    enable_binding: bool = True
    binding_strength: float = 0.1
    include_position: bool = True
    class_count: int = 10
    min_input_norm: float = 1e-4
    init_seed: int | None = 7
    enable_spatial_attention: bool = True
    attention_gain_strength: float = 0.35
    attention_center_bias: float = 0.2
    enable_lateral_inhibition: bool = True
    inhibition_similarity_threshold: float = 0.9
    enable_adaptive_capacity: bool = True
    max_pool_sizes: list[int] | None = None
    capacity_growth_factor: float = 1.35
    capacity_abstention_window: int = 24
    capacity_abstention_threshold: float = 0.35
    capacity_prune_interval: int = 256
    capacity_prune_min_presentations: int = 96
    capacity_prune_max_fire_count: int = 0
    feedback_strength: float = 0.0
    predictive: PredictiveConfig | None = None
    stdp: STDPConfig | None = None

    def __post_init__(self) -> None:
        height, width, channels = (int(value) for value in self.image_size)
        if height <= 0 or width <= 0 or channels <= 0:
            raise ValueError("image_size must contain positive integers.")
        self.image_size = (height, width, channels)
        self.init_seed = None if self.init_seed is None else int(self.init_seed)
        self.enable_spatial_attention = bool(self.enable_spatial_attention)
        self.enable_lateral_inhibition = bool(self.enable_lateral_inhibition)
        self.enable_adaptive_capacity = bool(self.enable_adaptive_capacity)
        self.attention_gain_strength = float(max(0.0, self.attention_gain_strength))
        self.attention_center_bias = float(max(0.0, self.attention_center_bias))
        self.inhibition_similarity_threshold = float(self.inhibition_similarity_threshold)
        self.capacity_growth_factor = float(max(1.1, self.capacity_growth_factor))
        self.capacity_abstention_window = int(max(4, self.capacity_abstention_window))
        self.capacity_abstention_threshold = float(
            max(0.0, min(1.0, self.capacity_abstention_threshold))
        )
        self.capacity_prune_interval = int(max(0, self.capacity_prune_interval))
        self.capacity_prune_min_presentations = int(max(1, self.capacity_prune_min_presentations))
        self.capacity_prune_max_fire_count = int(max(0, self.capacity_prune_max_fire_count))
        self.feedback_strength = float(max(0.0, self.feedback_strength))

        if self.max_pool_sizes is None:
            growth_schedule = (1.5, 1.5, 1.5, 3.0)
            self.max_pool_sizes = [
                max(int(pool_size) + 4, int(round(float(pool_size) * growth_factor)))
                for pool_size, growth_factor in zip(self.pool_sizes, growth_schedule, strict=False)
            ]
        else:
            self.max_pool_sizes = [int(size) for size in self.max_pool_sizes]

        expected_lengths = (
            len(self.patch_sizes),
            len(self.pool_sizes),
            len(self.concept_dims),
            len(self.thresholds),
            len(self.learning_rates),
            len(self.max_pool_sizes),
        )
        if any(length != self.num_layers for length in expected_lengths):
            raise ValueError("All per-layer lists must match num_layers.")
        if self.num_layers != 4:
            raise ValueError("VisualHierarchy currently supports exactly 4 layers.")
        if self.height % self.patch_size != 0 or self.width % self.patch_size != 0:
            raise ValueError("The first patch_size must evenly divide the image dimensions.")
        if any(max_size < pool_size for max_size, pool_size in zip(self.max_pool_sizes, self.pool_sizes, strict=False)):
            raise ValueError("Each max_pool_size must be greater than or equal to its pool_size.")
        if self.predictive is not None:
            self.predictive.num_levels = int(self.predictive.num_levels)
            self.predictive.settling_steps = int(max(1, self.predictive.settling_steps))
            if self.predictive.num_levels != self.num_layers:
                raise ValueError("predictive.num_levels must match the hierarchy num_layers.")
        if self.stdp is not None:
            self.stdp.tau_plus = float(self.stdp.tau_plus)
            self.stdp.tau_minus = float(self.stdp.tau_minus)
            self.stdp.A_plus = float(self.stdp.A_plus)
            self.stdp.A_minus = float(self.stdp.A_minus)

    @property
    def height(self) -> int:
        return int(self.image_size[0])

    @property
    def width(self) -> int:
        return int(self.image_size[1])

    @property
    def channels(self) -> int:
        return int(self.image_size[2])

    @property
    def flat_dim(self) -> int:
        return self.height * self.width * self.channels

    @property
    def patch_size(self) -> int:
        return int(self.patch_sizes[0])

    @property
    def position_dim(self) -> int:
        return 2 if self.include_position else 0

    @property
    def l1_input_dim(self) -> int:
        return (self.patch_size * self.patch_size * self.channels) + self.position_dim


__all__ = ["HierarchyConfig"]
