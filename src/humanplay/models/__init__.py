from .base import Model, Random, binomial_nll
from .levelk import CognitiveHierarchy, LevelK
from .nash import Nash
from .qre import QRE


def classic_models():
    """Fresh, unfitted instances of the four classic players plus the random floor."""
    return [Random(), Nash(), QRE(), LevelK(K=3), CognitiveHierarchy()]


__all__ = ["Model", "Random", "Nash", "QRE", "LevelK", "CognitiveHierarchy",
           "binomial_nll", "classic_models"]
