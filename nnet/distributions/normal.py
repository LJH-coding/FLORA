# PyTorch
import torch

class Normal(torch.distributions.Normal):

    def __init__(self, loc, scale, validate_args=None):
        super(Normal, self).__init__(loc=loc, scale=scale, validate_args=validate_args)

    def mode(self):
        return super(Normal, self).mode