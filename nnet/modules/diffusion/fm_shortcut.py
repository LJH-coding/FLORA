import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np
import math

# condition on h_t, z_t (from linear as a simple prior), I_t, t
# output z_t (from flow-matching, as a strong prior)

class ShortCut(nn.Module):
    def __init__(
        self,
        in_features,
        stoch_size,
        discrete,
        hidden_size,
        time_emb_dim: int = 128,
        d_min: int = 3, # 2^-{d_min}
    ):
        super().__init__()
        self.in_features = in_features
        self.stoch_size = stoch_size
        self.discrete = discrete
        self.hidden_size = hidden_size
        self.time_emb_dim = time_emb_dim

        self.step_size = [i for i in range(0, d_min+1)]
        self.d_min = d_min

        self.backbone = nn.Sequential(
            nn.Linear(in_features + time_emb_dim * 2 + self.stoch_size * self.discrete, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, self.stoch_size * self.discrete),
        )

        self.time_embedder = TimestepEmbedder(time_emb_dim)
        self.stepsize_embedder = nn.Embedding(len(self.step_size), time_emb_dim)

    def forward_net(self, condition, I_t, t, d):
        """
        I_t: (B, L, stoch * discrete)      // noisy latent
        condition: (B, L, hidden_size + stoch * discrete)  // h_t, z_t
        t: (B, L, 1) // current timestep
        d: (B, L, 1) // stepsize
        """
        B, L, D = I_t.shape
        I_t = I_t.reshape(B*L, -1)
        condition = condition.reshape(B*L, -1)
        t = t.reshape(B*L, -1)
        t_emb = self.time_embedder(t) # (B*L, time_emb_dim)

        d = d.reshape(B*L).long()
        d_emb = self.stepsize_embedder(d) # (B*L, time_emb_dim)

        x = torch.cat([condition, I_t, t_emb, d_emb], dim=-1)

        out = self.backbone(x)
        out = out.reshape(B, L, -1)
        return out # (B, L, stoch * discrete)

    def denoiser(self, condition, I_t, t, d):
        B, L, D = condition.shape
        out = self.forward_net(condition, I_t, t, d)
        out = out.reshape(B, L, self.stoch_size, self.discrete)
        return torch.softmax(out, dim=-1).reshape(B, L, self.stoch_size * self.discrete)
    
    def loss(self, posts, condition):
        """
        posts["stoch"]: (B, L, stoch * discrete)
        posts["logits"]: (B, L, stoch * discrete)
        condition: (B, L, hidden_size + stoch * discrete)  // h_t, z_t
        """

        B, L, _ = posts["stoch"].shape
        device = posts["stoch"].device

        # 其中 1/4 部分傳入 shortcut loss, 3/4 部分傳入 flow matching loss
        n_shortcut = B // 4
        perm = torch.randperm(B, device=device)
        idx_shortcut = perm[:n_shortcut]
        idx_fm = perm[n_shortcut:]

        total = 0.0
        denom = 0

        if idx_fm.numel() > 0:
            posts_fm = {"stoch": posts["stoch"][idx_fm]}
            fm_loss = self.flowmatching_loss(posts_fm, condition[idx_fm])  # (B_fm, L)
            total = total + fm_loss.sum()
            denom += fm_loss.numel()

        if idx_shortcut.numel() > 0:
            posts_sc = {"stoch": posts["stoch"][idx_shortcut]}
            sc_loss = self.shortcut_loss(posts_sc, condition[idx_shortcut])  # (B_sc, L)
            total = total + sc_loss.sum()
            denom += sc_loss.numel()

        if denom == 0:
            return torch.tensor(0.0, device=device)
        return total / denom

    def flowmatching_loss(self, posts, condition):
        # Flow matching loss
        B, L, D = posts["stoch"].shape
        x_1 = posts["stoch"].detach()
        x_0 = torch.randn_like(x_1)
        device = x_1.device

        t = torch.rand(B, L, device=device).view(B, L, 1)
        d = torch.full((B, L, 1), self.d_min, device=device, dtype=torch.long)
        x_t = (1 - t) * x_0 + t * x_1  # B, L, stoch * discrete

        logits = self.forward_net(condition, x_t, t, d) # B, L, stoch * discrete

        token_ids = x_1.reshape(B*L, self.stoch_size, self.discrete).argmax(dim=-1)
        logits = logits.reshape(B*L*self.stoch_size, self.discrete)
        token_ids = token_ids.reshape(B*L*self.stoch_size)

        fm_loss = F.cross_entropy(logits, token_ids, reduction='none')  # (B*L*stoch,)
        fm_loss = fm_loss.reshape(B, L, self.stoch_size).mean(dim=-1)   # (B, L)

        w = 0.9 * t + 0.1
        fm_loss = fm_loss * w.squeeze(-1)

        return fm_loss
    
    def shortcut_loss(self, posts, condition):
        # Shortcut loss
        B, L, D = posts["stoch"].shape
        x_1 = posts["stoch"].detach()
        x_0 = torch.randn_like(x_1)
        device = x_1.device

        t = torch.rand(B, L, device=device).view(B, L, 1)
        # sample exponent per (b,t) from {0, ..., d_min-1}; half-step then is exponent + 1
        if self.d_min > 0:
            d = torch.randint(0, self.d_min, (B, L, 1), device=device, dtype=torch.long)
        else:
            d = torch.zeros((B, L, 1), device=device, dtype=torch.long)

        step = torch.pow(2.0, -d.float())
        t_sc = (t * (1.0 - step))
        t_mid = t_sc + step / 2.0

        x_t = (1 - t_sc) *  x_0 + t_sc * x_1

        d_half = d + 1

        with torch.no_grad():
            v1 = self._velocity(self.denoiser(condition, x_t, t_sc, d_half), x_t, t_sc)
            x_mid = x_t + v1 * (step / 2.0)

            v2 = self._velocity(self.denoiser(condition, x_mid, t_mid, d_half), x_mid, t_mid)
        
            v_tgt = ((v1 + v2) / 2).detach()

        v_pred = self._velocity(self.denoiser(condition, x_t, t_sc, d), x_t, t_sc)

        v_err = v_pred - v_tgt

        shortcut_loss = (v_err ** 2).reshape(B, L, -1).mean(dim=-1) # (B, L)

        w = 0.9 * t_sc + 0.1
        shortcut_loss = shortcut_loss * w.squeeze(-1)

        return shortcut_loss

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
            d_now = torch.full((B, L, 1), dt.item(), device=device)
            D_t = self.denoiser(condition, z, t_now, self._d_to_exp(d_now))
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
    
    # ── helpers ──────────────────────────────────────────────────
    def _make_t(self, val, B, L, device):
        return torch.full((B, L, 1), val, device=device)

    def _velocity(self, D, z, tau):
        """b = (D - z) / (1 - τ)"""
        return (D - z) / (1.0 - tau + 1e-5)

    def _d_to_exp(self, d):
        x = torch.round(-torch.log2(torch.clamp(d, min=1e-8)))
        return x.long().clamp(min=0, max=self.d_min)

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