"""Configuration for hierarchical visual feature learning."""

from __future__ import annotations

from dataclasses import dataclass, field


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

    def __post_init__(self) -> None:
        height, width, channels = (int(value) for value in self.image_size)
        if height <= 0 or width <= 0 or channels <= 0:
            raise ValueError("image_size must contain positive integers.")
        self.image_size = (height, width, channels)

        expected_lengths = (
            len(self.patch_sizes),
            len(self.pool_sizes),
            len(self.concept_dims),
            len(self.thresholds),
            len(self.learning_rates),
        )
        if any(length != self.num_layers for length in expected_lengths):
            raise ValueError("All per-layer lists must match num_layers.")
        if self.num_layers != 4:
            raise ValueError("VisualHierarchy currently supports exactly 4 layers.")
        if self.height % self.patch_size != 0 or self.width % self.patch_size != 0:
            raise ValueError("The first patch_size must evenly divide the image dimensions.")

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
