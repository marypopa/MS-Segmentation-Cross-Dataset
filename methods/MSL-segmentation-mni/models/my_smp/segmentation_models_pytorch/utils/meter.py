import numpy as np
import torch

class Meter(object):
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self):
        """Reset the meter to default settings."""
        pass

    def add(self, value):
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """
        pass

    def value(self):
        """Get the value of the meter in the current state."""
        pass


class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        # self.val = 0

    def add(self, values):
        self.sum += torch.sum(values)
        self.sum_sq = self.sum**2
        self.n += values.numel()
        self.mean = self.sum / self.n
        self.mean = self.mean.numpy().item()
        self.std = torch.sqrt(self.sum_sq/(self.n-1))
        self.std = self.std.numpy().item()
        if values.ndimension() == 0:
            values = values.unsqueeze(0)
        self.values = torch.cat((self.values, values))
    # def value(self):
    #     return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.mean = 0.0
        self.sum_sq = 0.0
        self.std = 0.0
        self.values = torch.tensor([])
        # self.var = 0.0
        # self.val = 0.0
        # self.mean = np.nan
        # self.mean_old = 0.0
        # self.m_s = 0.0
        # self.std = np.nan

