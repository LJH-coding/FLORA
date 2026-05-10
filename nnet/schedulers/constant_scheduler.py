# Neural Nets
from nnet.schedulers import Scheduler

class ConstantScheduler(Scheduler):

    def __init__(self, val):
        super(ConstantScheduler, self).__init__()

        # Scheduler Params
        self.val = val

    def get_val_step(self, step):
        return self.val