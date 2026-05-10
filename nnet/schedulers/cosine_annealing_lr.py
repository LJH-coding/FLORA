# Python
import math

# Neural Nets
from nnet.schedulers import Scheduler

class CosineAnnealingLR(Scheduler):

    def __init__(self, lr_max, T_max, eta_min=0):
        super(CosineAnnealingLR, self).__init__()
        
        # Scheduler Params
        self.lr_max = lr_max
        self.T_max = T_max
        self.eta_min = eta_min

    def get_val_step(self, step):
        if step == 0:
            return self.lr_max
        elif step <= self.T_max:
            return self.eta_min + (self.lr_max - self.eta_min) * (1 + math.cos(math.pi * step / self.T_max)) / 2
        else:
            return self.eta_min
