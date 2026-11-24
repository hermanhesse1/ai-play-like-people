from .base import Model, Random, binomial_nll
from .levelk import CognitiveHierarchy, LevelK
from .nash import Nash
from .qre import QRE


def classic_models():
    """Fresh, unfitted instances of the four classic players plus the random floor."""
    return [Random(), Nash(), QRE(), LevelK(K=3), CognitiveHierarchy()]


def all_models(neural: bool = True):
    """The classic players and, from stage 2, the neural network.

    The network needs PyTorch, so it is imported only when asked for; the
    classic models keep working without it.
    """
    models = classic_models()
    if neural:
        from .neural import NeuralNet
        models.append(NeuralNet())
    return models


__all__ = ["Model", "Random", "Nash", "QRE", "LevelK", "CognitiveHierarchy",
           "binomial_nll", "classic_models", "all_models"]
