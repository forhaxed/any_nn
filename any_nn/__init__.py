"""
any_nn - A collection of neural network training utilities.
"""

from .any_ema import AnyEMA
from .any_trainer import AnyTrainer

__version__ = "0.1.0"
__all__ = ["AnyEMA", "AnyTrainer", "__version__"]
