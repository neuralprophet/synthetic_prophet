import torch
from torch import Tensor
from .base_noise import BaseNoise


__all__ = ["GaussianNoise"]


class GaussianNoise(BaseNoise):
    """Gaussian noise generator.
    This class adds uncorrelated, additive white noise to your signal.

    Attributes
    ----------
    mean : float
        mean for the noise
    std : float
        standard deviation for the noise

    """

    def __init__(self, mean=0, std=1.0):
        self.vectorizable = True
        self.mean = mean
        self.std = std

    def sample_next(self, t:int, samples:torch.tensor, errors:torch.tensor)-> Tensor:
        return torch.normal(mean=self.mean, std=self.std, size=(1,))

    def sample_vectorized(self, time_vector:torch.tensor)-> Tensor:
        n_samples = len(time_vector)
        return torch.normal(mean=self.mean, std=self.std, size=(n_samples,))
