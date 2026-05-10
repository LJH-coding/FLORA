# PyTorch
import torch
import torch.nn as nn

class Scheduler(nn.Module):

    def __init__(self):
        super(Scheduler, self).__init__()

        # Model Step
        self.model_step = torch.tensor(0)

    def step(self):
        self.model_step += 1
        return self.get_val()

    def get_val(self):
        return self.get_val_step(self.model_step)

    def get_val_step(self, step):
        return None