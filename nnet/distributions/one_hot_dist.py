# PyTorch
import torch
from torch.nn import functional as F

class OneHotDist(torch.distributions.one_hot_categorical.OneHotCategoricalStraightThrough):

  def __init__(self, probs=None, logits=None, validate_args=None, uniform_mix=0.0, sampling_tmp=1.0):

    # Uniform Mix
    if uniform_mix > 0 and logits is not None:
      probs = F.softmax(logits / sampling_tmp, dim=-1)
      probs = (1 - uniform_mix) * probs + uniform_mix / probs.shape[-1]
      logits = torch.log(probs)
      super(OneHotDist, self).__init__(logits=logits, probs=None, validate_args=validate_args)
    elif sampling_tmp != 1.0 and logits is not None:
      probs = F.softmax(logits / sampling_tmp, dim=-1)
      logits = torch.log(probs)
      super(OneHotDist, self).__init__(logits=logits, probs=None, validate_args=validate_args)
    else:
      super(OneHotDist, self).__init__(logits=logits, probs=probs, validate_args=validate_args)

  def mode(self):
    mode = super(OneHotDist, self).mode
    return mode.detach() + (self.logits - self.logits.detach())