"""
Sleep-EDF HDP-HMM Package

Hierarchical Dirichlet Process Hidden Markov Models for unsupervised sleep staging.
"""

__version__ = "0.1.0"
__author__ = "Matteo Omizzolo"

from . import data
from . import models
from . import inference
from . import eval
from . import utils

__all__ = ["data", "models", "inference", "eval", "utils"]
