"""Bio-ARN 2.0 hyperparameters and configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bioarn.ensemble.config import EnsembleConfig
    from bioarn.hierarchy.config import HierarchyConfig


@dataclass
class SpikingConfig:
    """LIF neuron parameters."""
    beta: float = 0.9            # Membrane potential decay factor
    threshold: float = 1.0       # Spike threshold voltage
    reset: float = 0.0           # Reset potential after spike
    dt: float = 1.0              # Time step (ms)
    refractory_steps: int = 2    # Refractory period after spike


@dataclass
class MarginGateConfig:
    """Margin gate (honest abstention) parameters."""
    theta_margin: float = 0.5       # Minimum cosine similarity to fire
    theta_margin_lr: float = 0.001  # Threshold adaptation rate
    theta_resonance: float = 0.7    # Minimum match for resonance learning


@dataclass
class STDPConfig:
    """Spike-timing-dependent plasticity parameters."""

    tau_plus: float = 20.0
    tau_minus: float = 40.0
    A_plus: float = 0.05
    A_minus: float = 0.06

    def __post_init__(self) -> None:
        self.tau_plus = float(self.tau_plus)
        self.tau_minus = float(self.tau_minus)
        self.A_plus = float(self.A_plus)
        self.A_minus = float(self.A_minus)


@dataclass
class CCCConfig:
    """Concept Cell Cluster parameters."""
    input_dim: int = 784         # Dimensionality of input (e.g., 28×28 for MNIST)
    concept_dim: int = 256       # Dimensionality of concept direction vectors
    num_f1_features: int = 128   # Number of F1 feature neurons (competitive selection)
    f1_top_k: int = 32           # How many F1 features survive competition
    fast_lr: float = 1.0         # One-shot learning rate (immediate encoding)
    slow_lr: float = 0.01        # Hebbian tuning rate (gradual refinement)
    feedback_lr: float = 0.01    # Feedback weight learning rate
    max_pool_size: int = 1000    # Maximum number of CCCs in the pool
    stdp: STDPConfig | None = None

    def __post_init__(self) -> None:
        if isinstance(self.stdp, Mapping):
            self.stdp = STDPConfig(**self.stdp)


@dataclass
class SDMConfig:
    """Sparse Distributed Memory (Kanerva) parameters."""
    address_dim: int = 10000     # Dimensionality of binary address space
    hamming_radius: int = 451    # Retrieval radius (addresses within this distance)
    num_hard_locations: int = 1000  # Number of physical memory locations
    data_dim: int = 256          # Dimensionality of stored data vectors
    decay_rate: float = 0.999    # Association strength decay per step
    stdp_window: int = 20        # Temporal window for STDP (ms)


@dataclass
class PredictiveConfig:
    """Predictive coding engine parameters."""
    num_levels: int = 4          # Hierarchy depth
    gamma: float = 0.1           # State update rate from prediction errors
    eta: float = 0.01            # Weight learning rate (local Hebbian)
    precision_init: float = 1.0  # Initial precision weighting
    error_threshold: float = 0.01  # Below this, errors are suppressed (PCL)
    settling_steps: int = 6      # Iterative inference steps for resonance/settling


@dataclass
class GNWConfig:
    """Global Neuronal Workspace parameters."""
    capacity: int = 7            # Miller's Law: 7±2 items in working memory
    broadcast_gain: float = 2.0  # Amplification factor for broadcast CCCs
    fatigue_rate: float = 0.1    # How fast broadcast items fade
    fatigue_threshold: float = 0.3  # Below this, item exits GNW
    competition_temp: float = 1.0   # Softmax temperature for competitive selection
    concept_dim: int = 256       # Expected dimensionality of workspace concepts
    context_size: int = 128      # Extended context buffer size beyond the GNW slots
    context_decay: float = 0.95  # Decay factor for extended context items
    context_eviction_threshold: float = 0.05  # Forget items weaker than this
    context_update_rate: float = 0.2  # EMA rate for running context summary
    attention_heads: int = 4     # Number of spike-attention heads
    context_top_k: int = 5       # Max context items to retrieve per query
    recurrent_integration_rate: float = 0.1  # Leaky integration rate across time
    context_bias_gain: float = 0.35  # Additive bias strength during generation
    repetition_window: int = 20  # Context history window for repetition detection
    repetition_novelty_threshold: float = 0.8  # Inject novelty above this repetition score


@dataclass
class RewardConfig:
    """Reward and novelty system parameters."""
    intrinsic_scale: float = 1.0   # Scale of prediction error reward
    novelty_threshold: float = 3.0  # Prediction error magnitude for orienting response
    novelty_boost: float = 5.0     # Learning rate multiplier during novelty
    novelty_decay: float = 0.95    # How fast novelty boost fades
    curiosity_weight: float = 0.5  # Balance between exploitation and exploration


@dataclass
class BioARNConfig:
    """Master configuration for the full Bio-ARN 2.0 system."""
    spiking: SpikingConfig = field(default_factory=SpikingConfig)
    margin_gate: MarginGateConfig = field(default_factory=MarginGateConfig)
    ccc: CCCConfig = field(default_factory=CCCConfig)
    sdm: SDMConfig = field(default_factory=SDMConfig)
    predictive: PredictiveConfig = field(default_factory=PredictiveConfig)
    gnw: GNWConfig = field(default_factory=GNWConfig)
    workspace: GNWConfig | None = None
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Optional module configs — when set, enable hierarchy preprocessing and
    # ensemble voting in BioARNCore; when None those modules are bypassed.
    hierarchy: HierarchyConfig | None = None
    ensemble: EnsembleConfig | None = None

    # Global settings
    device: str = "cpu"          # "cpu" or "cuda"
    dtype: str = "float32"       # Default tensor dtype
    seed: int = 42               # Random seed for reproducibility
