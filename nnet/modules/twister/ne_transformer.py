import math
import numpy as np
import re
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd


def cal_fan(m):
    if isinstance(m, nn.Linear):
        in_num = m.in_features
        out_num = m.out_features
        init_type = "trunc_normal"
    elif isinstance(m, nn.RMSNorm):
        in_num, out_num = None, None
        init_type = "ones"
    elif isinstance(m, nn.Conv2d):
        space = m.kernel_size[0] * m.kernel_size[1]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        init_type = "trunc_normal"
    elif isinstance(m, nn.Conv1d):
        space = m.kernel_size[0] * m.kernel_size[0]
        in_num = space * m.in_channels
        out_num = space * m.out_channels
        init_type = "trunc_normal"
    else:
        in_num, out_num, init_type = None, None, None
    return in_num, out_num, init_type

def weight_init_(m, fan_type="in", scale=1.0):
    in_num, out_num, init_type = cal_fan(m)
    if scale == 0.0:
        m.weight.data.fill_(0.0)
    elif init_type == "trunc_normal":
        fan = {"avg": (in_num + out_num)/2, "in": in_num, "out": out_num}[fan_type]
        std = 1.1368 * np.sqrt(1 / fan) * scale
        nn.init.trunc_normal_(
            m.weight.data, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )
    elif init_type == "ones":
        m.weight.data.fill_(1.0 * scale)

    if hasattr(m, "bias") and hasattr(m.bias, "data"):
        m.bias.data.fill_(0.0)


class NEDreamerTransformer(nn.Module):
    """Causal Transformer for NE-Dreamer: predicts encoder embeddings from RSSM feat.
    
    Takes a sequence of feat (RSSM features) and optionally actions, predicts
    encoder embeddings using causal attention with configurable output heads:
    - head_same: predict embed[t] from feat[t] (same-timestep grounding)
    - head_next_k: predict embed[t+k] from feat[t] (multi-token prediction, k = 1..predict_horizon)
    
    Architecture:
    - With actions: Interleaved [f0, a0, f1, a1, ...] tokens
      - Same-timestep: predict at feat token positions
      - Next-timestep: predict at action token positions
    - Without actions: [f0, f1, f2, ...] tokens
      - Same-timestep: predict at each position for same embed
      - Next-timestep: predict at each position for next embed
    - Causal masking ensures prediction at time t only sees up to time t
    - Multi-token prediction: separate head for each horizon k
    """
    def __init__(self, feat_dim, output_dim, action_dim, hidden_dim=256, num_layers=2, num_heads=4, 
                 max_seq_len=128, dropout=0.0, use_actions=True, act_discrete=False, act_classes=None,
                 use_same=True, use_next=True, predict_horizon=1):
        super().__init__()
        self.feat_dim = feat_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.use_actions = use_actions
        self.act_discrete = act_discrete
        self.use_same = use_same
        self.use_next = use_next
        self.predict_horizon = predict_horizon
        
        assert use_same or use_next, "At least one of use_same or use_next must be True"
        assert predict_horizon >= 1, "predict_horizon must be at least 1"
        
        # Token embeddings for feat
        self.f_embed = nn.Linear(feat_dim, hidden_dim)
        
        # Action embeddings (only if use_actions=True)
        self.use_embedding = False  # Track if using nn.Embedding vs nn.Linear
        if use_actions:
            if act_discrete and act_classes is not None:
                self.a_embed = nn.Embedding(act_classes, hidden_dim)
                self.use_embedding = True
            else:
                self.a_embed = nn.Sequential(
                    nn.Linear(action_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                )
            # Positional embeddings for interleaved sequence (2 * max_seq_len)
            self.pos_embed = nn.Parameter(torch.zeros(1, 2 * max_seq_len, hidden_dim))
        else:
            self.a_embed = None
            # Positional embeddings for feat-only sequence (max_seq_len)
            self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Transformer encoder with causal masking
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads for next-timestep prediction: predict embed[t+k] from h[t]
        # Multi-token prediction: separate head for each horizon k = 1, 2, ..., predict_horizon
        if use_next:
            self.heads_next = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, output_dim),
                )
                for _ in range(predict_horizon)
            ])
        else:
            self.heads_next = None
        
        # Output head for same-timestep prediction: predict embed[t] from h[t]
        if use_same:
            self.head_same = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.head_same = None
        
        self.apply(weight_init_)
        # Re-init pos_embed after weight_init_
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def _generate_causal_mask(self, seq_len, device):
        """Generate causal attention mask (upper triangular = -inf)."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, feat, actions=None):
        """
        Args:
            feat: RSSM features (B, T, feat_dim) - features from f[0] to f[T-1]
            actions: Actions (B, T, A) or (B, T) if discrete - actions from a[0] to a[T-1]
                     Only used if use_actions=True
        
        Returns:
            If dual_head=True:
                e_hat_same: Predicted same-timestep embeddings (B, T, output_dim) - predictions for e[0..T-1]
                e_hat_next: Predicted next-timestep embeddings (B, T-1, output_dim) - predictions for e[1..T-1]
            If dual_head=False:
                e_hat_next: Predicted next-timestep embeddings (B, T-1, output_dim) - predictions for e[1..T-1]
        """
        B, T, _ = feat.shape
        device = feat.device
        
        # Embed feat tokens
        tok_f = self.f_embed(feat)  # (B, T, H)
        
        if self.use_actions:
            # With actions: interleave [f0, a0, f1, a1, ...]
            assert actions is not None, "Actions required when use_actions=True"
            
            if self.use_embedding:
                # Using nn.Embedding: need integer indices
                if actions.dtype in (torch.float16, torch.float32, torch.float64):
                    # One-hot encoded: convert to indices
                    action_indices = actions.argmax(dim=-1).long()
                else:
                    # Already integer indices
                    action_indices = actions.long()
                tok_a = self.a_embed(action_indices)  # (B, T, H)
            else:
                # Using nn.Linear: need float tensors (one-hot or continuous)
                tok_a = self.a_embed(actions.float())  # (B, T, H)
            
            # Interleave: [f0, a0, f1, a1, ..., f_{T-1}, a_{T-1}]
            # Shape: (B, 2*T, H)
            tokens = torch.stack([tok_f, tok_a], dim=2).reshape(B, 2 * T, -1)
            
            # Add positional embeddings
            tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
            
            # Causal mask
            causal_mask = self._generate_causal_mask(tokens.size(1), device)
            
            # Transformer forward
            h = self.transformer(tokens, mask=causal_mask)  # (B, 2*T, H)
            
            # Extract hidden states at feat token positions for same-timestep prediction
            # f[t] is at index 2*t in interleaved sequence
            # For t=0..T-1, indices are: 0, 2, 4, ..., 2*(T-1)
            f_token_indices = torch.arange(0, 2 * T, 2, device=device)  # (T,)
            h_same = h[:, f_token_indices, :]  # (B, T, H)
            
            # Extract hidden states at action token positions for next-timestep prediction
            # To predict e[t+1], we use the hidden state after seeing a[t]
            # a[t] is at index 2*t+1 in interleaved sequence
            # For predicting e[1..T-1] from a[0..T-2], indices are: 1, 3, 5, ..., 2*(T-1)-1
            a_token_indices = torch.arange(1, 2 * T - 1, 2, device=device)  # (T-1,)
            h_next = h[:, a_token_indices, :]  # (B, T-1, H)
        else:
            # Without actions: just [f0, f1, f2, ...]
            tokens = tok_f  # (B, T, H)
            
            # Add positional embeddings
            tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
            
            # Causal mask
            causal_mask = self._generate_causal_mask(tokens.size(1), device)
            
            # Transformer forward
            h = self.transformer(tokens, mask=causal_mask)  # (B, T, H)
            
            # For same-timestep: h[t] predicts e[t]
            h_same = h  # (B, T, H)
            
            # For next-timestep: h[t] predicts e[t+1], so use h[0..T-2] to predict e[1..T-1]
            h_next = h[:, :-1, :]  # (B, T-1, H)
        
        # Project to predicted embeddings based on enabled heads
        e_hat_same = None
        e_hat_next_list = None
        
        if self.use_same:
            e_hat_same = self.head_same(h_same)  # (B, T, output_dim)
        
        if self.use_next:
            # Multi-token prediction: each head predicts a different horizon
            # heads_next[k] predicts embed[t+k+1] from h[t]
            # For horizon k, we need h[0..T-k-1] to predict embed[k+1..T]
            e_hat_next_list = []
            B, T_h, _ = h_next.shape  # h_next is (B, T-1, H) for use_actions, or needs to be computed per horizon
            
            for k in range(self.predict_horizon):
                # For horizon k+1 (1-indexed), we predict embed[t+k+1] from h[t]
                # Valid predictions: t can range from 0 to T-k-2 (so we have targets at t+k+1)
                if k < T_h:
                    h_for_k = h_next[:, :T_h - k, :]  # (B, T-1-k, H)
                    e_hat_k = self.heads_next[k](h_for_k)  # (B, T-1-k, output_dim)
                    e_hat_next_list.append(e_hat_k)
                else:
                    # Not enough sequence length for this horizon
                    e_hat_next_list.append(None)
        
        # Return based on which heads are enabled
        if self.use_same and self.use_next:
            return e_hat_same, e_hat_next_list
        elif self.use_same:
            return e_hat_same
        else:  # use_next only
            return e_hat_next_list