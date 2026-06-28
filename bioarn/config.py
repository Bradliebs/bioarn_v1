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
class LayerwiseTrainConfig:
    """Greedy layer-wise Hebbian pre-training parameters."""

    enabled: bool = False
    samples_per_layer: int = 2000
    passes_per_layer: int = 3
    lr_schedule: list[float] = field(
        default_factory=lambda: [0.01, 0.005, 0.003, 0.002, 0.001]
    )
    freeze_after_training: bool = True

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.samples_per_layer = int(max(1, self.samples_per_layer))
        self.passes_per_layer = int(max(1, self.passes_per_layer))
        normalized_schedule = [float(max(0.0, lr)) for lr in self.lr_schedule]
        self.lr_schedule = normalized_schedule or [0.01]
        self.freeze_after_training = bool(self.freeze_after_training)


@dataclass
class ConvCCCConfig:
    """Convolutional CCC parameters."""

    in_channels: int = 3
    spatial_size: int = 32
    num_conv_features: int = 64
    num_conv_layers: int = 3
    conv_hidden_channels: tuple[int, ...] = (32, 64)
    conv_kernel_sizes: tuple[int, ...] = (5, 3, 3)
    spatial_grid: int = 4
    concept_dim: int = 0
    f1_top_k: int = 64
    fast_lr: float = 1.0
    slow_lr: float = 0.01
    feedback_lr: float = 0.01
    conv_hebbian_lr: float = 0.0025
    hebbian_batch_size: int = 32
    conv_competitive_k: int = 8
    spatial_top_k: int = 4
    conv_weight_norm: float = 1.0
    enable_local_contrast_norm: bool = True
    contrast_kernel_size: int = 5
    response_norm_eps: float = 1e-4
    feature_pool_avg_mix: float = 0.25
    hebbian_oja_decay: float = 0.05
    filter_decorrelation: float = 0.02
    softhebb_enabled: bool = False
    softhebb_gamma: float = 4.0
    softhebb_beta: float = 2.0
    softhebb_theta_decay: float = 0.99
    layerwise_train: LayerwiseTrainConfig = field(default_factory=LayerwiseTrainConfig)
    max_pool_size: int = 200
    max_growth_factor: float = 3.0
    consolidation_strength: float = 0.0
    lock_threshold: float = 0.8

    def feature_channels(self) -> tuple[int, ...]:
        if self.num_conv_layers <= 1:
            return (self.num_conv_features,)
        hidden_channels = list(self.conv_hidden_channels)
        if not hidden_channels:
            hidden_channels = [max(32, self.num_conv_features)]
        while len(hidden_channels) < max(0, self.num_conv_layers - 1):
            hidden_channels.append(max(hidden_channels[-1], self.num_conv_features))
        return tuple(hidden_channels[: self.num_conv_layers - 1]) + (self.num_conv_features,)

    def __post_init__(self) -> None:
        if isinstance(self.layerwise_train, Mapping):
            self.layerwise_train = LayerwiseTrainConfig(**self.layerwise_train)
        self.in_channels = int(max(1, self.in_channels))
        self.spatial_size = int(max(1, self.spatial_size))
        self.num_conv_features = int(max(1, self.num_conv_features))
        self.num_conv_layers = int(max(1, self.num_conv_layers))
        hidden_channels = tuple(int(max(1, channel)) for channel in self.conv_hidden_channels)
        if not hidden_channels:
            hidden_channels = (max(32, self.num_conv_features),)
        if len(hidden_channels) < max(0, self.num_conv_layers - 1):
            expanded = list(hidden_channels)
            while len(expanded) < max(0, self.num_conv_layers - 1):
                expanded.append(max(expanded[-1], self.num_conv_features))
            hidden_channels = tuple(expanded)
        self.conv_hidden_channels = hidden_channels[: max(1, self.num_conv_layers - 1)]
        kernel_sizes = tuple(int(max(1, kernel)) for kernel in self.conv_kernel_sizes)
        if not kernel_sizes:
            kernel_sizes = (5,)
        if len(kernel_sizes) < self.num_conv_layers:
            kernel_sizes = kernel_sizes + (kernel_sizes[-1],) * (self.num_conv_layers - len(kernel_sizes))
        normalized_kernel_sizes = []
        for kernel_size in kernel_sizes[: self.num_conv_layers]:
            normalized_kernel_sizes.append(kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        self.conv_kernel_sizes = tuple(normalized_kernel_sizes)
        self.spatial_grid = int(max(1, self.spatial_grid))
        if int(self.concept_dim) <= 0:
            self.concept_dim = sum(self.feature_channels()) * self.spatial_grid * self.spatial_grid
        else:
            self.concept_dim = int(self.concept_dim)
        self.f1_top_k = int(max(1, self.f1_top_k))
        self.conv_hebbian_lr = float(max(0.0, self.conv_hebbian_lr))
        self.hebbian_batch_size = int(max(1, self.hebbian_batch_size))
        self.conv_competitive_k = int(max(1, self.conv_competitive_k))
        self.spatial_top_k = int(max(1, self.spatial_top_k))
        self.conv_weight_norm = float(max(1e-6, self.conv_weight_norm))
        self.contrast_kernel_size = int(max(1, self.contrast_kernel_size))
        if self.contrast_kernel_size % 2 == 0:
            self.contrast_kernel_size += 1
        self.response_norm_eps = float(max(1e-8, self.response_norm_eps))
        self.feature_pool_avg_mix = float(min(max(self.feature_pool_avg_mix, 0.0), 1.0))
        self.hebbian_oja_decay = float(max(0.0, self.hebbian_oja_decay))
        self.filter_decorrelation = float(max(0.0, self.filter_decorrelation))
        self.softhebb_enabled = bool(self.softhebb_enabled)
        self.softhebb_gamma = float(max(1e-6, self.softhebb_gamma))
        self.softhebb_beta = float(max(1.0, self.softhebb_beta))
        self.softhebb_theta_decay = float(min(max(self.softhebb_theta_decay, 0.0), 0.999999))
        self.max_pool_size = int(max(1, self.max_pool_size))
        self.max_growth_factor = float(max(1.0, self.max_growth_factor))
        self.consolidation_strength = float(max(0.0, self.consolidation_strength))
        self.lock_threshold = float(min(max(self.lock_threshold, 0.0), 1.0))


def deep_cifar_config() -> ConvCCCConfig:
    """Deeper CIFAR-oriented convolutional CCC preset with layer-wise pre-training."""

    return ConvCCCConfig(
        in_channels=3,
        spatial_size=32,
        num_conv_features=384,
        num_conv_layers=5,
        conv_hidden_channels=(96, 128, 192, 256),
        conv_kernel_sizes=(5, 3, 3, 3, 3),
        spatial_grid=4,
        f1_top_k=64,
        conv_hebbian_lr=0.005,
        hebbian_batch_size=32,
        conv_competitive_k=16,
        spatial_top_k=8,
        layerwise_train=LayerwiseTrainConfig(
            enabled=True,
            samples_per_layer=2000,
            passes_per_layer=3,
            lr_schedule=[0.01, 0.005, 0.003, 0.002, 0.001],
            freeze_after_training=True,
        ),
    )


@dataclass
class AugmentationConfig:
    """Configuration for training-time vision augmentation."""

    enabled: bool = False
    random_flip: bool = True
    random_crop: bool = True
    color_jitter: bool = False
    cutout: bool = False
    cutout_size: int = 8
    augmentation_factor: int = 2

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.random_flip = bool(self.random_flip)
        self.random_crop = bool(self.random_crop)
        self.color_jitter = bool(self.color_jitter)
        self.cutout = bool(self.cutout)
        self.cutout_size = int(max(1, self.cutout_size))
        self.augmentation_factor = int(max(1, self.augmentation_factor))


@dataclass
class WhiteningConfig:
    """Offline whitening parameters for vision streams."""

    enabled: bool = False
    epsilon: float = 1e-5
    n_fit_samples: int = 5000
    n_components: int | None = None

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.epsilon = float(max(1e-12, self.epsilon))
        self.n_fit_samples = int(max(2, self.n_fit_samples))
        if self.n_components is not None:
            self.n_components = int(max(1, self.n_components))


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
class MultimodalFusionConfig:
    """Configuration for GNW-mediated multimodal fusion."""

    vision_enabled: bool = True
    audio_enabled: bool = True
    temporal_enabled: bool = True
    concept_dim: int = 256
    vision_pool_size: int = 200
    audio_pool_size: int = 100
    workspace_size: int = 16
    cross_modal_weight: float = 0.5
    agreement_threshold: float = 0.6
    margin_threshold: float = 0.35
    learning_rate: float = 0.05
    audio: AudioConfig = field(default_factory=AudioConfig)
    audio_hierarchy: AudioHierarchyConfig = field(default_factory=AudioHierarchyConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    workspace: GNWConfig | None = None
    precision: PrecisionConfig | None = None
    sdm: SDMConfig | None = None

    def __post_init__(self) -> None:
        if isinstance(self.audio, Mapping):
            self.audio = AudioConfig(**self.audio)
        if isinstance(self.audio_hierarchy, Mapping):
            self.audio_hierarchy = AudioHierarchyConfig(**self.audio_hierarchy)
        if isinstance(self.temporal, Mapping):
            self.temporal = TemporalConfig(**self.temporal)
        if isinstance(self.workspace, Mapping):
            self.workspace = GNWConfig(**self.workspace)
        if isinstance(self.precision, Mapping):
            self.precision = PrecisionConfig(**self.precision)
        if isinstance(self.sdm, Mapping):
            self.sdm = SDMConfig(**self.sdm)

        self.vision_enabled = bool(self.vision_enabled)
        self.audio_enabled = bool(self.audio_enabled)
        self.temporal_enabled = bool(self.temporal_enabled)
        self.concept_dim = int(max(1, self.concept_dim))
        self.vision_pool_size = int(max(1, self.vision_pool_size))
        self.audio_pool_size = int(max(1, self.audio_pool_size))
        self.workspace_size = int(max(1, self.workspace_size))
        self.cross_modal_weight = float(min(max(self.cross_modal_weight, 0.0), 1.0))
        self.agreement_threshold = float(min(max(self.agreement_threshold, 0.0), 1.0))
        self.margin_threshold = float(min(max(self.margin_threshold, 0.0), 1.0))
        self.learning_rate = float(max(0.0, self.learning_rate))

        self.temporal.concept_dim = int(self.concept_dim)
        self.audio_hierarchy.n_mels = int(self.audio.n_mels)

        if self.workspace is None:
            self.workspace = GNWConfig(
                capacity=self.workspace_size,
                concept_dim=self.concept_dim,
            )
        else:
            self.workspace.capacity = int(self.workspace_size)
            self.workspace.concept_dim = int(self.concept_dim)

        precision_pool = max(
            8,
            self.vision_pool_size + self.audio_pool_size + max(1, self.concept_dim // 8),
        )
        if self.precision is None:
            self.precision = PrecisionConfig(
                enabled=True,
                pool_size=precision_pool,
                entropy_window=64,
                precision_alpha=6.0,
                precision_threshold=0.4,
                min_precision=0.15,
                max_precision=1.0,
                lateral_error_weight=1.0,
                hierarchy_error_weight=0.25,
                external_signal_decay=0.7,
                surprise_gain=1.0,
            )
        else:
            self.precision.enabled = True
            self.precision.pool_size = int(precision_pool)

        if self.sdm is None:
            self.sdm = SDMConfig(
                address_dim=max(256, self.concept_dim * 4),
                hamming_radius=max(16, self.concept_dim // 4),
                num_hard_locations=max(128, self.concept_dim * 4),
                data_dim=self.concept_dim,
                decay_rate=0.999,
                stdp_window=max(4, self.temporal.context_window),
            )
        else:
            self.sdm.data_dim = int(self.concept_dim)
            self.sdm.stdp_window = int(max(1, self.temporal.context_window))


@dataclass
class MultimodalTrainConfig:
    """Streaming training configuration for multimodal fusion."""

    fusion: MultimodalFusionConfig = field(default_factory=MultimodalFusionConfig)
    num_samples: int = 96
    num_passes: int = 2
    num_classes: int = 4
    vision_size: int = 16
    shuffle: bool = True
    seed: int = 0
    learning_rate_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.fusion, Mapping):
            self.fusion = MultimodalFusionConfig(**self.fusion)
        self.num_samples = int(max(1, self.num_samples))
        self.num_passes = int(max(1, self.num_passes))
        self.num_classes = int(max(2, self.num_classes))
        self.vision_size = int(max(8, self.vision_size))
        self.shuffle = bool(self.shuffle)
        self.seed = int(self.seed)
        self.learning_rate_multiplier = float(max(0.0, self.learning_rate_multiplier))


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
