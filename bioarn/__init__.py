"""Bio-ARN 2.0: Brain-inspired, low-power, multi-modal generative architecture."""

from bioarn._version import __version__
from bioarn.config import BioARNConfig
from bioarn.loop import SensorimotorLoop
from bioarn.system import BioARNCore

__all__ = ["BioARNConfig", "BioARNCore", "SensorimotorLoop", "__version__"]
