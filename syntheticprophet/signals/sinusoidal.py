import numpy as np
from .base_signal import BaseSignal
from torch import Tensor
from typing import Callable
import torch


__all__ = ["Sinusoidal"]


class Sinusoidal(BaseSignal):
    """Signal generator for harmonic (sinusoidal) waves.

    Parameters
    ----------
    amplitude : number (default 1.0)
        Amplitude of the harmonic series
    frequency : number (default 1.0)
        Frequency of the harmonic series
    ftype : function (default np.sin)
        Harmonic function

    """

    def __init__(self,
                 amplitude:float=1.0,
                 frequency:float=1.0,
                 ftype=torch.sin):
        super().__init__(vectorizable=True)
        self.amplitude:float = amplitude
        self.ftype:Callable[[Tensor],Tensor]= ftype
        self.frequency:float = frequency

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
        return self.amplitude * self.ftype(2 * np.pi * self.frequency * time)

    def sample_vectorized(self, time_vector:Tensor)->Tensor:
        """Sample entire series based off of time vector

        Parameters
        ----------
        time_vector : array-like
            Timestamps for signal generation

        Returns
        -------
        array-like
            sampled signal for time vector

        """
        if self.vectorizable is True:
            signal = self.amplitude * self.ftype(
                2 * np.pi * self.frequency * time_vector.clone().detach()
            )
            return signal
        else:
            raise ValueError("Signal type not vectorizable")
