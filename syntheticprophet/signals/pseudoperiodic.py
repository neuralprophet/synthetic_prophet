import torch
from torch import Tensor
from typing import Callable
from .base_signal import BaseSignal

__all__ = ["PseudoPeriodic"]


class PseudoPeriodic(BaseSignal):
    """Signal generator for pseudoeriodic waves.

    The wave's amplitude and frequency have some stochasticity that
    can be set manually.

    Parameters
    ----------
    amplitude : number (default 1.0)
        Amplitude of the harmonic series
    frequency : number (default 1.0)
        Frequency of the harmonic series
    ampSD : number (default 0.1)
        Amplitude standard deviation
    freqSD : number (default 0.1)
        Frequency standard deviation
    ftype : function(default np.sin)
        Harmonic function

    """

    def __init__(
        self, amplitude:float=1.0,
            frequency:int=100,
            ampSD:float=0.1,
            freqSD:float=0.4,
            ftype:Callable[[Tensor],Tensor]=torch.sin
    ):
        self.vectorizable = True
        self.amplitude = amplitude
        self.frequency = frequency
        self.freqSD = freqSD
        self.ampSD = ampSD
        self.ftype = ftype

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
        freq_val = torch.normal(mean=self.frequency, std=self.freqSD, size=(1,))
        amplitude_val = torch.normal(mean=self.amplitude, std=self.ampSD, size=(1,))
        return float(amplitude_val * torch.sin(freq_val * time))

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
        n_samples = len(time_vector)
        freq_arr = torch.normal(
            mean=self.frequency, std=self.freqSD, size=(n_samples,)
        )
        amp_arr = torch.normal(mean=self.amplitude, std=self.ampSD, size=(n_samples,))
        signal = torch.mul(amp_arr, self.ftype(torch.mul(freq_arr, time_vector.clone().detach())))
        return signal
