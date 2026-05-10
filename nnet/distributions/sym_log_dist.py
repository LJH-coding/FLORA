# PyTorch
import torch

# NeuralNets
from nnet import modules

class SymLogDist:

    def __init__(self, mode, reinterpreted_batch_ndims, eps=1e-8):
        self._mode = mode
        self.dims = tuple([-x for x in range(1, reinterpreted_batch_ndims + 1)])
        self.eps = eps

    def mode(self):
        return modules.sym_exp(self._mode)
    
    def mean(self):
        return modules.sym_exp(self._mode)
    
    def log_prob(self, value):

        # assert
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)

        # L2 dist
        distance = (self._mode - modules.sym_log(value)) ** 2

        # eps
        distance = torch.where(distance < self.eps, 0, distance)

        # Reduction
        loss = distance.sum(self.dims)

        return - loss
    
