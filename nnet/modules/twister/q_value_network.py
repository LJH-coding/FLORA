# PyTorch
import torch
import torch.nn as nn

# NeuralNets
from nnet import modules
from nnet import distributions

class QValueBase(nn.Module):

    def __init__(
        self, 
        hidden_size=512, 
        act_fun=nn.SiLU, 
        num_mlp_layers=2, 
        feat_size=32*32+512, 
        weight_init="dreamerv3_normal", 
        bias_init="zeros", 
        norm={"class": "LayerNorm", "params": {"eps": 1e-3}}, 
        bins=255,
        dist_weight_init="zeros",
        dist_bias_init="zeros",
    ):
        super(QValueBase, self).__init__()

        self.mlp = modules.MultiLayerPerceptron(dim_input=feat_size, dim_layers=[hidden_size for _ in range(num_mlp_layers)], act_fun=act_fun, weight_init=weight_init, bias_init=bias_init, norm=norm, bias=norm is None)
        self.linear_proj = modules.Linear(hidden_size, bins, weight_init=dist_weight_init, bias_init=dist_bias_init)

    def forward(self, x):

        # MLP Layers
        x = self.mlp(x)

        # Output Proj
        x = self.linear_proj(x)

        # Distributional Q head (two-hot targets via SymLogDiscreteDist.log_prob)
        return distributions.SymLogDiscreteDist(logits=x, reinterpreted_batch_ndims=1, low=-20, high=20)

class QValueNetwork(nn.Module):
    def __init__(
        self, 
        hidden_size=512, 
        act_fun=nn.SiLU, 
        num_mlp_layers=2, 
        num_actions=6,
        feat_size=32*32+512, 
        weight_init="dreamerv3_normal", 
        bias_init="zeros", 
        norm={"class": "LayerNorm", "params": {"eps": 1e-3}}, 
        bins=255,
        dist_weight_init="zeros",
        dist_bias_init="zeros",
    ):
        super(QValueNetwork, self).__init__()
        self.q1 = QValueBase(
            hidden_size=hidden_size, 
            act_fun=act_fun, 
            num_mlp_layers=num_mlp_layers, 
            feat_size=feat_size + num_actions, 
            weight_init=weight_init, 
            bias_init=bias_init, 
            norm=norm, 
            bins=bins,
            dist_weight_init=dist_weight_init,
            dist_bias_init=dist_bias_init,
        )
        self.q2 = QValueBase(
            hidden_size=hidden_size, 
            act_fun=act_fun, 
            num_mlp_layers=num_mlp_layers, 
            feat_size=feat_size + num_actions, 
            weight_init=weight_init, 
            bias_init=bias_init, 
            norm=norm, 
            bins=bins,
            dist_weight_init=dist_weight_init,
            dist_bias_init=dist_bias_init,
        )

    def forward(self, state, action):
        # given [st, at] output q value

        # Q1 / Q2 distributional outputs
        q1 = self.q1(torch.cat([state, action], dim=-1))
        q2 = self.q2(torch.cat([state, action], dim=-1))

        return q1, q2