# PyTorch
import torch

# NeuralNets
from nnet import utils

class Dataset(torch.utils.data.Dataset):

    def __init__(self, num_workers=0, batch_size=8, collate_fn=utils.CollateDefault(), root="datasets", shuffle=True, persistent_workers=False):
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.root = root
        self.shuffle = shuffle
        self.persistent_workers = persistent_workers if self.num_workers > 0 else False