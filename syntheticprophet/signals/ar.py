import torch
from torch import Tensor
from .base_signal import BaseSignal
from typing import List

__all__ = ["AutoRegressive"]


class AutoRegressive(BaseSignal):
    """Sample generator for autoregressive (AR) signals.

    Generates time series with an autogressive lag defined by the number of parameters in ar_param.
    NOTE: Only use this for regularly sampled signals

    Parameters
    ----------
    ar_param : list (default [None])
        Parameter of the AR(p) process
        [phi_1, phi_2, phi_3, .... phi_p]
    sigma : float (default 1.0)
        Standard deviation of the signal
    start_value : list (default [None])
        Starting value of the AR(p) process

    """

    def __init__(self,
                 ar_param:List=[None],
                 sigma:float=0.5,
                 start_value:List=[None]):
        super().__init__(vectorizable=False)
        #if start_value is None:
        #    start_value = torch.Tensor()
        # if ar_param is None:
        #     ar_param = torch.Tensor()
        # if isinstance(ar_param, List):
        #     ar_param = torch.tensor(ar_param)
        # ar_param.reverse()
        # self.ar_param = torch.tensor(ar_param)
        # self.sigma = sigma
        #
        # if start_value[0] is None:
        #     self.start_value = torch.tensor([0 for i in range(len(ar_param))])
        # else:
        #     if len(start_value) != len(ar_param):
        #         raise ValueError("AR parameters do not match starting value")
        #     else:
        #         self.start_value = start_value
        # if isinstance(self.start_value, List):
        #     self.start_value = torch.tensor(self.start_value)
        # self.previous_value = self.start_value

        ar_param.reverse()
        self.ar_param = ar_param
        self.sigma = sigma
        if start_value[0] is None:
            self.start_value = [0 for i in range(len(ar_param))]
        else:
            if len(start_value) != len(ar_param):
                raise ValueError("AR parameters do not match starting value")
            else:
                self.start_value = start_value
        self.previous_value = self.start_value



    def sample_next(self, time:int, samples:Tensor, errors:Tensor)->float:
        """Sample a single time point

        Parameters
        ----------
        time : number
            Time at which a sample was required

        Returns
        -------
        ar_value : float
            sampled signal for time t
        """
        #ar_value = torch.tensor([
        #    self.previous_value[i] * self.ar_param[i] for i in range(len(self.ar_param))
        #])
        # assert(len(self.previous_value)==len(self.ar_param))
        # ar_value = torch.mul(self.previous_value, self.ar_param)

        ar_value = [self.previous_value[i] * self.ar_param[i] for i in range(len(self.ar_param))]

        noise = torch.normal(mean=0.0, std=self.sigma, size=(1,))
        ar_value = torch.sum(ar_value) + noise
        self.previous_value = self.previous_value[1:] + ar_value


        return ar_value
