import torch
from torch import Tensor
from .base_signal import BaseSignal

__all__ = ["CAR"]


class CAR(BaseSignal):
    """Signal generator for continuously autoregressive (CAR) signals.

    Parameters
    ----------
    ar_param : number (default 1.0)
        Parameter of the AR(1) process
    sigma : number (default 1.0)
        Standard deviation of the signal
    start_value : number (default 0.0)
        Starting value of the AR process

    """

    def __init__(self, ar_param:float=1.0, sigma:float=0.5, start_value:float=0.01):
        self.vectorizable = False
        self.ar_param = ar_param
        self.sigma = sigma
        self.start_value = start_value
        self.previous_value = None
        self.previous_time = None

    def sample_next(self, time:int, samples:Tensor, errors:Tensor)->float:
        """Sample a single time point

        Parameters
        ----------
        time : number
            Time at which a sample was required

        Returns
        -------
        float
            sampled signal for time t

        """
        if self.previous_value is None:
            output = self.start_value
        else:
            time_diff = time - self.previous_time
            noise = torch.normal(mean=0.0, std=1.0, size=(1,))
            if isinstance(self.ar_param, float):
                self.ar_param = torch.tensor([self.ar_param])
            if isinstance(time_diff, float):
                time_diff = torch.tensor([time_diff])

            output = (
                torch.pow(self.ar_param, time_diff)
            ) * self.previous_value + self.sigma * torch.sqrt(
                1 - torch.pow(self.ar_param, time_diff)
            ) * noise
        self.previous_time = time
        self.previous_value = output
        return output
