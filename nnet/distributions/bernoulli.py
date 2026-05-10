# PyTorch
import torch
import torch.nn.functional as F

class Bernoulli(torch.distributions.Bernoulli):

    def __init__(self, probs=None, logits=None, validate_args=None):
        super(Bernoulli, self).__init__(probs=probs, logits=logits, validate_args=validate_args)

    def log_prob(self, x):
        logits = self.logits
        log_probs0 = - F.softplus(logits)
        log_probs1 = - F.softplus(-logits)

        return log_probs0 * (1-x) + log_probs1 * x
    
    @property
    def mode(self):
        mode = (self.probs > 0.5).to(self.probs)
        return mode