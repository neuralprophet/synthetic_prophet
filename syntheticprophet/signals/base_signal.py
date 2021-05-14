__all__ = []
from torch import Tensor


class BaseSignal:
    """BaseSignal class

    Signature for all signal classes.

    """

    def __init__(self, vectorizable:bool=False):
        self.vectorizable = vectorizable

    def sample_next(self, time:int, samples:Tensor, errors:Tensor)->float:
        """Samples next point based on history of samples and errors

        Parameters
        ----------
        time : int
            time
        samples : array-like
            all samples taken so far
        errors : array-like
            all errors sampled so far

        Returns
        -------
        float
            sampled signal for time t

        """
        raise NotImplementedError

    def sample_vectorized(self, time_vector:Tensor)->Tensor:
        """Samples for all time points in input

        Parameters
        ----------
        time_vector : array like
            all time stamps to be sampled

        Returns
        -------
        float
            sampled signal for time t

        """
        raise NotImplementedError
