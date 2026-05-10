import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np
import math

# condition on h_t, z_t (from linear as a simple prior), I_t, t
# output z_t (from flow-matching, more strong prior)

class FM(nn.Module):
    def __init__(
        self,
        in_features,
        stoch_size,
        discrete,
        hidden_size,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        self.in_features = in_features
        self.stoch_size = stoch_size
        self.discrete = discrete
        self.hidden_size = hidden_size
        self.time_emb_dim = time_emb_dim

        self.backbone = nn.Sequential(
            nn.Linear(in_features + time_emb_dim + self.stoch_size * self.discrete, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, self.stoch_size * self.discrete),
        )

        self.time_embedder = TimestepEmbedder(time_emb_dim)


    def forward_net(self, condition, I_t, t):
        """
        I_t: (B, L, stoch * discrete)      // noisy latent
        condition: (B, L, hidden_size + stoch * discrete)  // h_t, z_t
        t: (B, L, 1) // current timestep
        """
        B, L, D = I_t.shape
        I_t = I_t.reshape(B*L, -1)
        condition = condition.reshape(B*L, -1)
        t = t.reshape(B*L, -1)
        t_emb = self.time_embedder(t) # (B*L, time_emb_dim)

        x = torch.cat([condition, I_t, t_emb], dim=-1)

        out = self.backbone(x)
        out = out.reshape(B, L, -1)
        return out # (B, L, stoch * discrete)

    def denoiser(self, condition, I_t, t):
        B, L, D = condition.shape
        out = self.forward_net(condition, I_t, t)
        out = out.reshape(B, L, self.stoch_size, self.discrete)
        return torch.softmax(out, dim=-1).reshape(B, L, self.stoch_size * self.discrete)
    
    def loss(self, posts, condition):
        """
        posts["stoch"]: (B, L, stoch * discrete)
        posts["logits"]: (B, L, stoch * discrete)
        condition: (B, L, hidden_size + stoch * discrete)  // h_t, z_t
        """

        B, L, D = posts["stoch"].shape
        x_1 = posts["stoch"].detach()
        t = torch.rand(B, L, device=condition.device).view(B, L, 1)
        x_0 = torch.randn_like(x_1)
        x_t = (1 - t) * x_0 + t * x_1  # B, L, stoch * discrete
        logits = self.forward_net(condition, x_t, t) # B, L, stoch * discrete
        
        x_1 = x_1.reshape(B*L, self.stoch_size, self.discrete)
        token_ids = x_1.argmax(dim=-1)
        logits = logits.reshape(B*L*self.stoch_size, self.discrete)
        token_ids = token_ids.reshape(B*L*self.stoch_size)

        loss = F.cross_entropy(logits, token_ids).mean()

        return loss

    @torch.no_grad()
    def sample(self, condition, n_steps: int = 1):
        B, L, D = condition.shape
        device  = condition.device

        z  = torch.randn(B, L, self.stoch_size * self.discrete, device=device)

        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        dt = ts[1] - ts[0]

        for i in range(n_steps):
            t_now = ts[i].item()
            t_now = torch.full((B, L, 1), t_now, device=device)
            D_t = self.denoiser(condition, z, t_now)
            if i == n_steps-1:
                z = D_t
                break
            v = (D_t - z) / (1.0 - t_now + 1e-5)
            z = z + dt * v             # Euler step

        z = z.reshape(B, L, self.stoch_size, self.discrete)
        z = z.argmax(dim=-1)
        z = F.one_hot(z, num_classes=self.discrete)
        z = z.reshape(B, L, self.stoch_size * self.discrete).float()
        return z # (B, L, stoch * discrete)

class LearnableLossWeighting(nn.Module):
    def __init__(self, cond_dim, hidden_dim=128):
        super().__init__()
        self.s_embed = TimestepEmbedder(cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, t):
        return self.mlp(self.s_embed(t)).squeeze(-1)   # (B,)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            - math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding,
                 torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.squeeze(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb