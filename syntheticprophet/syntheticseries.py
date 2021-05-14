import torch
from torch import Tensor
from .noise.base_noise import BaseNoise
from .signals.base_signal import BaseSignal
from typing import Tuple

__all__ = ["SyntheticSeries"]


class SyntheticSeries:
    """A SyntheticSeries object is the main interface from which to sample synthetic time series.
    Provide at least a signal generator; a noise generator is optional.
    It is recommended to set the sampling frequency.

    Parameters
    ----------
    signal_generator : Signal object
        signal object for synthetic time series
    noise_generator : Noise object
        noise object for synthetic time series

    """

    def __init__(self, signal_generator:BaseSignal, noise_generator:BaseNoise=None):
        self.signal_generator = signal_generator
        self.noise_generator = noise_generator

    def sample(self, time_vector:Tensor)->Tuple[Tensor, Tensor, Tensor]:
        """Samples from the specified SyntheticSeries.

        Parameters
        ----------
        time_vector : tensor
            Times at which to generate a sample

        Returns
        -------
        samples, signals, errors, : tuple (tensor, tensor, tensor)
            Returns samples, and the signals and errors they were constructed from
        """

        # Vectorize if possible
        if (
            self.signal_generator.vectorizable
            and not self.noise_generator is None
            and self.noise_generator.vectorizable
        ):
            signals = self.signal_generator.sample_vectorized(time_vector)
            errors = self.noise_generator.sample_vectorized(time_vector)
            samples = signals + errors
        elif self.signal_generator.vectorizable and self.noise_generator is None:
            signals = self.signal_generator.sample_vectorized(time_vector)
            errors = torch.zeros(len(time_vector))
            samples = signals
        else:
            n_samples = len(time_vector)
            samples = torch.zeros(n_samples)  # Signal and errors combined
            signals = torch.zeros(n_samples)  # Signal samples
            errors = torch.zeros(n_samples)  # Handle errors separately
            times = torch.arange(n_samples)

            # Sample iteratively, while providing access to all previously sampled steps
            for i in range(n_samples):
                # Get time
                t = time_vector[i]
                # Sample error
                if not self.noise_generator is None:
                    errors[i] = self.noise_generator.sample_next(
                        t, samples[: i - 1], errors[: i - 1]
                    )

                # Sample signal
                signal = self.signal_generator.sample_next(
                    t, samples[: i - 1], errors[: i - 1]
                )
                signals[i] = signal

                # Compound signal and noise
                samples[i] = signals[i] + errors[i]

        # Return both times and samples, as well as signals and errors
        return samples, signals, errors
