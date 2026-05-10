# PyTorch
import torch

# NeuralNets
from nnet import datasets
from nnet import utils

class VoidDataset(datasets.Dataset):

    def __init__(self, num_steps=100):
        super(VoidDataset, self).__init__(batch_size=1, collate_fn=utils.CollateFn(inputs_params=[{ "axis": 0 }], targets_params=[{ "axis": 0 }]), shuffle=False, root=None)
        self.num_steps = num_steps

    def __getitem__(self, n):
        return torch.tensor(n),

    def __len__(self):
        return self.num_steps
