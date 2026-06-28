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
class PrecisionConfig:
    """Precision-weighted predictive processing parameters."""

    enabled: bool = False
    pool_size: int = 100
    entropy_window: int = 100
    precision_alpha: float = 5.0
    precision_threshold: float = 0.5
    min_precision: float = 0.1
    max_precision: float = 1.0
    lateral_error_weight: float = 0.35
    hierarchy_error_weight: float = 0.2
    external_signal_decay: float = 0.85
    surprise_gain: float = 1.5

    def __post_init__(self) -> None:
        self.pool_size = int(max(1, self.pool_size))
        self.entropy_window = int(max(1, self.entropy_window))
        self.precision_alpha = float(self.precision_alpha)
        self.precision_threshold = float(self.precision_threshold)
        self.min_precision = float(max(0.0, self.min_precision))
        self.max_precision = float(max(self.min_precision, self.max_precision))
        self.lateral_error_weight = float(max(0.0, self.lateral_error_weight))
        self.hierarchy_error_weight = float(max(0.0, self.hierarchy_error_weight))
        self.external_signal_decay = float(min(max(self.external_signal_decay, 0.0), 0.999))
        self.surprise_gain = float(max(0.0, self.surprise_gain))


@dataclass
class LateralPredictionConfig:
    """Sparse lateral predictive coding parameters for CCC pools."""

    enabled: bool = False
    max_neighbors: int = 8
    hebbian_lr: float = 0.05
    anti_hebbian_lr: float = 0.02
    min_weight: float = 0.1
    max_weight: float = 2.5
    refresh_interval: int = 16
    prediction_threshold: float = 0.1
    surprise_gain: float = 1.5

    def __post_init__(self) -> None:
        self.max_neighbors = int(max(1, self.max_neighbors))
        self.hebbian_lr = float(max(0.0, self.hebbian_lr))
        self.anti_hebbian_lr = float(max(0.0, self.anti_hebbian_lr))
        self.min_weight = float(max(0.0, self.min_weight))
        self.max_weight = float(max(self.min_weight, self.max_weight))
        self.refresh_interval = int(max(1, self.refresh_interval))
        self.prediction_threshold = float(min(max(self.prediction_threshold, 0.0), 1.0))
        self.surprise_gain = float(max(0.0, self.surprise_gain))


@dataclass
class CCCConfig:
    """Concept Cell Cluster parameters."""
    input_dim: int = 784         # Dimensionality of input (e.g., 28×28 for MNIST)
    concept_dim: int = 256       # Dimensionality of concept direction vectors
    num_f1_features: int = 128   # Number of F1 feature neurons (competitive selection)
    f1_top_k: int = 32           # How many F1 features survive competition
    freeze_f1_after: int = 0     # Freeze shared F1 and enable task adapters after N samples
    f1_adapter_dim: int = 16     # Bottleneck width for residual task adapters
    fast_lr: float = 1.0         # One-shot learning rate (immediate encoding)
    slow_lr: float = 0.01        # Hebbian tuning rate (gradual refinement)
    feedback_lr: float = 0.01    # Feedback weight learning rate
    max_pool_size: int = 1000    # Maximum number of CCCs in the pool
    max_growth_factor: float = 3.0  # Allow the pool to grow beyond its initial size
    consolidation_strength: float = 0.0  # Penalize updates to highly active CCCs
    lock_threshold: float = 0.8  # Lock CCC when importance exceeds this
    protection_growth_rate: float = 0.1  # Grow soft protection for valuable CCCs
    protection_decay_rate: float = 0.01  # Slowly release stale protected CCCs
    replay_interval: int = 64    # Samples between concept replay sweeps
    enable_elastic_protection: bool = False
    enable_replay: bool = False
    enable_eviction: bool = False
    stdp: STDPConfig | None = None
    precision: PrecisionConfig | None = None
    lateral_prediction: LateralPredictionConfig | None = None

    def __post_init__(self) -> None:
        if isinstance(self.stdp, Mapping):
            self.stdp = STDPConfig(**self.stdp)
        if isinstance(self.precision, Mapping):
            self.precision = PrecisionConfig(**self.precision)
        if isinstance(self.lateral_prediction, Mapping):
            self.lateral_prediction = LateralPredictionConfig(**self.lateral_prediction)
        self.freeze_f1_after = int(max(0, self.freeze_f1_after))
        self.f1_adapter_dim = int(max(1, self.f1_adapter_dim))
        self.max_pool_size = int(max(1, self.max_pool_size))
        self.max_growth_factor = float(max(1.0, self.max_growth_factor))
        self.consolidation_strength = float(max(0.0, self.consolidation_strength))
        self.lock_threshold = float(max(0.0, self.lock_threshold))
        self.protection_growth_rate = float(min(max(self.protection_growth_rate, 0.0), 1.0))
        self.protection_decay_rate = float(min(max(self.protection_decay_rate, 0.0), 1.0))
        self.replay_interval = int(max(0, self.replay_interval))


@dataclass
class ConvCCCConfig:
    """Convolutional CCC parameters."""

    in_channels: int = 3
    spatial_size: int = 32
    num_conv_features: int = 64
    num_conv_layers: int = 3
    conv_hidden_channels: tuple[int, int] = (32, 64)
    spatial_grid: int = 4
    concept_dim: int = 0
    f1_top_k: int = 64
    fast_lr: float = 1.0
    slow_lr: float = 0.01
    feedback_lr: float = 0.01
    conv_hebbian_lr: float = 0.0025
    conv_competitive_k: int = 8
    spatial_top_k: int = 4
    conv_weight_norm: float = 1.0
    max_pool_size: int = 200
    max_growth_factor: float = 3.0
    consolidation_strength: float = 0.0
    lock_threshold: float = 0.8

    def feature_channels(self) -> tuple[int, ...]:
        hidden1 = int(self.conv_hidden_channels[0])
        hidden2 = int(self.conv_hidden_channels[1])
        if self.num_conv_layers <= 1:
            return (self.num_conv_features,)
        if self.num_conv_layers == 2:
            return (hidden1, self.num_conv_features)
        return (hidden1, hidden2, self.num_conv_features)

    def __post_init__(self) -> None:
        self.in_channels = int(max(1, self.in_channels))
        self.spatial_size = int(max(1, self.spatial_size))
        self.num_conv_features = int(max(1, self.num_conv_features))
        self.num_conv_layers = int(min(max(1, self.num_conv_layers), 3))
        hidden_channels = tuple(int(max(1, channel)) for channel in self.conv_hidden_channels)
        if not hidden_channels:
            hidden_channels = (32, 64)
        if len(hidden_channels) == 1:
            hidden_channels = (hidden_channels[0], max(hidden_channels[0], self.num_conv_features))
        self.conv_hidden_channels = (hidden_channels[0], hidden_channels[1])
        self.spatial_grid = int(max(1, self.spatial_grid))
        if int(self.concept_dim) <= 0:
            self.concept_dim = sum(self.feature_channels()) * self.spatial_grid * self.spatial_grid
        else:
            self.concept_dim = int(self.concept_dim)
        self.f1_top_k = int(max(1, self.f1_top_k))
        self.conv_hebbian_lr = float(max(0.0, self.conv_hebbian_lr))
        self.conv_competitive_k = int(max(1, self.conv_competitive_k))
        self.spatial_top_k = int(max(1, self.spatial_top_k))
        self.conv_weight_norm = float(max(1e-6, self.conv_weight_norm))
        self.max_pool_size = int(max(1, self.max_pool_size))
        self.max_growth_factor = float(max(1.0, self.max_growth_factor))
        self.consolidation_strength = float(max(0.0, self.consolidation_strength))
        self.lock_threshold = float(min(max(self.lock_threshold, 0.0), 1.0))


@dataclass
class AudioConfig:
    """Audio preprocessing parameters."""

    sample_rate: int = 16000
    n_mels: int = 40
    n_fft: int = 512
    hop_length: int = 160
    max_duration_ms: int = 1000

    def __post_init__(self) -> None:
        self.sample_rate = int(max(1, self.sample_rate))
        self.n_mels = int(max(1, self.n_mels))
        self.n_fft = int(max(16, self.n_fft))
        self.hop_length = int(max(1, self.hop_length))
        self.max_duration_ms = int(max(1, self.max_duration_ms))

    @property
    def max_samples(self) -> int:
        return max(1, int(round(self.sample_rate * (self.max_duration_ms / 1000.0))))

    @property
    def max_frames(self) -> int:
        return 1 + (self.max_samples // self.hop_length)


@dataclass
class AudioHierarchyConfig:
    """Configuration for the auditory A1 → A2 → Belt hierarchy."""

    n_mels: int = 40
    a1_channels: int = 32
    a2_channels: int = 64
    belt_dim: int = 256
    temporal_kernel: int = 5
    init_seed: int = 7

    def __post_init__(self) -> None:
        self.n_mels = int(max(1, self.n_mels))
        self.a1_channels = int(max(1, self.a1_channels))
        self.a2_channels = int(max(1, self.a2_channels))
        self.belt_dim = int(max(1, self.belt_dim))
        self.temporal_kernel = int(max(1, self.temporal_kernel))
        if self.temporal_kernel % 2 == 0:
            self.temporal_kernel += 1
        self.init_seed = int(self.init_seed)


@dataclass
class AudioTrainConfig:
    """Online audio-training parameters."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    hierarchy: AudioHierarchyConfig = field(default_factory=AudioHierarchyConfig)
    max_pool_size: int = 128
    max_growth_factor: float = 2.0
    margin_threshold: float = 0.95
    learning_rate: float = 0.01
    num_train_samples: int = 300
    num_test_samples: int = 120
    num_passes: int = 2
    interleave_classes: bool = True
    use_batched: bool = True
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if isinstance(self.audio, Mapping):
            self.audio = AudioConfig(**self.audio)
        if isinstance(self.hierarchy, Mapping):
            self.hierarchy = AudioHierarchyConfig(**self.hierarchy)
        self.max_pool_size = int(max(1, self.max_pool_size))
        self.max_growth_factor = float(max(1.0, self.max_growth_factor))
        self.margin_threshold = float(min(max(self.margin_threshold, 0.0), 1.0))
        self.learning_rate = float(max(0.0, self.learning_rate))
        self.num_train_samples = int(max(1, self.num_train_samples))
        self.num_test_samples = int(max(1, self.num_test_samples))
        self.num_passes = int(max(1, self.num_passes))
        self.interleave_classes = bool(self.interleave_classes)
        self.use_batched = bool(self.use_batched)
        self.seed = int(self.seed)
        self.device = str(self.device)
        if self.hierarchy.n_mels != self.audio.n_mels:
            self.hierarchy.n_mels = int(self.audio.n_mels)


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
    gamma: float = 0.1           # Prediction-error gain for settling or learning gates
    eta: float = 0.01            # Weight learning rate (local Hebbian)
    precision_init: float = 1.0  # Initial precision weighting
    error_threshold: float = 0.01  # Below this, errors are suppressed (PCL)
    settling_steps: int = 6      # Iterative inference steps for resonance/settling
    mode: str = "error_gating"   # Predictive mode: error_gating or settling


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
class AssociativeMemoryConfig:
    """Configuration for the associative memory engine."""

    capacity: int = 500
    concept_dim: int = 256
    input_dim: int = 768
    top_k_retrieval: int = 5
    auto_consolidate_interval: int = 100
    lock_important: bool = True
    importance_threshold: float = 0.8
    use_workspace: bool = True
    use_precision: bool = True

    def __post_init__(self) -> None:
        self.capacity = int(max(1, self.capacity))
        self.concept_dim = int(max(1, self.concept_dim))
        self.input_dim = int(max(1, self.input_dim))
        self.top_k_retrieval = int(max(1, min(self.top_k_retrieval, self.capacity)))
        self.auto_consolidate_interval = int(max(0, self.auto_consolidate_interval))
        self.lock_important = bool(self.lock_important)
        self.importance_threshold = float(min(max(self.importance_threshold, 0.0), 1.0))
        self.use_workspace = bool(self.use_workspace)
        self.use_precision = bool(self.use_precision)


@dataclass
class TemporalConfig:
    """Temporal STDP layer parameters."""

    context_window: int = 8
    concept_dim: int = 256
    stdp_tau_plus: float = 20.0
    stdp_tau_minus: float = 40.0
    stdp_lr: float = 0.01
    prediction_threshold: float = 0.3

    def __post_init__(self) -> None:
        self.context_window = int(max(1, self.context_window))
        self.concept_dim = int(max(1, self.concept_dim))
        self.stdp_tau_plus = float(max(1e-6, self.stdp_tau_plus))
        self.stdp_tau_minus = float(max(1e-6, self.stdp_tau_minus))
        self.stdp_lr = float(max(0.0, self.stdp_lr))
        self.prediction_threshold = float(min(max(self.prediction_threshold, 0.0), 1.0))


@dataclass
class TemporalTrainConfig:
    """Streaming temporal-learning parameters."""

    frames_per_sequence: int = 8
    num_sequences: int = 200
    concept_dim: int = 256
    max_pool_size: int = 100
    use_workspace_context: bool = True
    frame_shape: tuple[int, int] = (16, 16)
    seed: int = 0
    margin_threshold: float = 0.35
    prediction_top_k: int = 8
    context_gain: float = 0.2
    causal_violation_rate: float = 0.2
    workspace: GNWConfig | None = None
    stdp: STDPConfig | None = None
    temporal: TemporalConfig | None = None

    def __post_init__(self) -> None:
        if isinstance(self.workspace, Mapping):
            self.workspace = GNWConfig(**self.workspace)
        if isinstance(self.stdp, Mapping):
            self.stdp = STDPConfig(**self.stdp)
        if isinstance(self.temporal, Mapping):
            self.temporal = TemporalConfig(**self.temporal)
        self.frames_per_sequence = int(max(2, self.frames_per_sequence))
        self.num_sequences = int(max(1, self.num_sequences))
        self.concept_dim = int(max(1, self.concept_dim))
        self.max_pool_size = int(max(1, self.max_pool_size))
        height = int(max(1, self.frame_shape[0]))
        width = int(max(1, self.frame_shape[1]))
        self.frame_shape = (height, width)
        self.seed = int(self.seed)
        self.margin_threshold = float(min(max(self.margin_threshold, 0.0), 1.0))
        self.prediction_top_k = int(max(1, min(self.prediction_top_k, self.concept_dim)))
        self.context_gain = float(max(0.0, self.context_gain))
        self.causal_violation_rate = float(min(max(self.causal_violation_rate, 0.0), 1.0))
        if self.temporal is None:
            self.temporal = TemporalConfig(
                context_window=self.frames_per_sequence,
                concept_dim=self.concept_dim,
            )
        else:
            self.temporal.context_window = int(max(1, self.temporal.context_window))
            self.temporal.concept_dim = int(max(1, self.concept_dim))


@dataclass
class RewardConfig:
    """Reward and novelty system parameters."""
    intrinsic_scale: float = 1.0   # Scale of prediction error reward
    novelty_threshold: float = 3.0  # Prediction error magnitude for orienting response
    novelty_boost: float = 5.0     # Learning rate multiplier during novelty
    novelty_decay: float = 0.95    # How fast novelty boost fades
    curiosity_weight: float = 0.5  # Balance between exploitation and exploration


@dataclass
class WorldModelConfig:
    """Configuration for the Bio-ARN RL world model."""

    observation_dim: int = 4
    concept_dim: int = 64
    max_pool_size: int = 50
    num_actions: int = 2
    curiosity_weight: float = 0.5
    prediction_lr: float = 0.05
    use_precision: bool = True

    def __post_init__(self) -> None:
        self.observation_dim = int(max(1, self.observation_dim))
        self.concept_dim = int(max(1, self.concept_dim))
        self.max_pool_size = int(max(2, self.max_pool_size))
        self.num_actions = int(max(1, self.num_actions))
        self.curiosity_weight = float(max(0.0, self.curiosity_weight))
        self.prediction_lr = float(max(1e-4, self.prediction_lr))
        self.use_precision = bool(self.use_precision)


@dataclass
class AgentConfig:
    """Configuration for the concept-based Bio-ARN RL agent."""

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    curiosity_bonus: float = 0.3
    reward_discount: float = 0.99

    def __post_init__(self) -> None:
        self.epsilon_start = float(min(max(self.epsilon_start, 0.0), 1.0))
        self.epsilon_end = float(min(max(self.epsilon_end, 0.0), 1.0))
        self.epsilon_decay = float(min(max(self.epsilon_decay, 0.0), 1.0))
        self.curiosity_bonus = float(max(0.0, self.curiosity_bonus))
        self.reward_discount = float(min(max(self.reward_discount, 0.0), 1.0))


@dataclass
class RLTrainConfig:
    """Configuration for training Bio-ARN on lightweight RL environments."""

    env_name: str = "cartpole"
    num_episodes: int = 500
    max_steps_per_episode: int = 500
    world_model: WorldModelConfig | None = None
    agent: AgentConfig | None = None

    def __post_init__(self) -> None:
        if isinstance(self.world_model, Mapping):
            self.world_model = WorldModelConfig(**self.world_model)
        if isinstance(self.agent, Mapping):
            self.agent = AgentConfig(**self.agent)
        if self.world_model is None:
            self.world_model = WorldModelConfig()
        if self.agent is None:
            self.agent = AgentConfig()
        self.env_name = str(self.env_name).strip().lower()
        self.num_episodes = int(max(1, self.num_episodes))
        self.max_steps_per_episode = int(max(1, self.max_steps_per_episode))


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
