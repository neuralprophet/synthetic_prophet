import torch
from torch import Tensor
from .base_noise import BaseNoise


__all__ = ["RedNoise"]


class RedNoise(BaseNoise):
    """Red noise generator.
    This class adds correlated (red) noise to your signal.

    Attributes
    ----------
    mean : float
        mean for the noise
    std : float
        standard deviation for the noise
    tau : float
        ?
    start_value : float
        ?

    """

    def __init__(self, mean:float=0, std:float=1.0, tau:float=0.2, start_value:float=0):
        self.vectorizable = False
        self.mean = mean
        self.std = std
        self.start_value = torch.tensor(start_value)
        self.tau = tau
        self.previous_value: Tensor = None
        self.previous_time = None

    def sample_next(self, t:int, samples:torch.tensor, errors:torch.tensor)->Tensor:
        if self.previous_time is None:
            red_noise = self.start_value
        else:
            time_diff = t - self.previous_time
            wnoise = torch.normal(mean=self.mean, std=self.std, size=(1,))
            red_noise = (self.tau / (self.tau + time_diff)) * (
                time_diff * wnoise + self.previous_value
            )
        self.previous_time = t
        self.previous_value = red_noise
        return red_noise
