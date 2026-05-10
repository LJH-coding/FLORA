import torch
import torch.nn as nn
import torch.nn.functional as F

from nnet import distributions

class Tokenizer(nn.Module):
    """Tokenizer with Binary Spherical Quantization"""
    
    def __init__(self, 
                encoder,
                decoder,
                quantizer, 
                image_channels,
                image_size,
                num_tokens):
        
        super().__init__()
        
        # Encoder & Decoder
        self.encoder = encoder
        self.decoder = decoder
        
        self.quantizer = quantizer

        self.num_tokens = num_tokens
        self.embed_dim = quantizer.dim

        _ = torch.zeros((1,) + (image_channels,) + tuple(image_size))
        self.shape = self.encoder(_).shape # (1, latent_dim, H', W')
        self.project_in = nn.Linear(self.shape[-3] * self.shape[-2] * self.shape[-1], self.quantizer.dim * self.num_tokens)
        self.project_out = nn.Linear(self.quantizer.dim * self.num_tokens, self.shape[-3] * self.shape[-2] * self.shape[-1])
        
    def encode(self, x):
        x = x.clone()
        """Encode images to binary latent codes"""
        # x: (B, T, C, H, W) or (B, C, H, W)
        z = self.encoder(x)  # (B, T, latent_dim, H', W') or (B, latent_dim, H', W')

        shape = z.shape # (B, T, latent_dim, H', W') or (B, latent_dim, H', W')

        z = z.reshape((-1,) + shape[-3:]) # (N, latent_dim, H', W')
        z = z.reshape((z.shape[0], -1)) # (N, latent_dim * H' * W')
        z = self.project_in(z) # (N, num_tokens * embed_dim)
        z = z.reshape((-1, self.num_tokens, self.embed_dim)) # (N, num_tokens, embed_dim)

        z_q, info, quantize_loss = self.quantizer(z) # (N, num_tokens, embed_dim)
        info = info.reshape(shape[:-3] + (self.num_tokens,)) # (B, T, num_tokens) or (B, num_tokens)

        z_q = z_q.reshape(shape[:-3] + z_q.shape[-2:]) # (B, T, num_tokens, embed_dim) or (B, num_tokens, embed_dim)

        return {
            'z_q': z_q,
            'vq_loss': quantize_loss,
            'info': info,
        }

    def get_logits(self, z_q, logits_type="hard"):
        z_q = z_q.clone()
        original_shape = z_q.shape
        z_q = z_q.reshape((-1, self.embed_dim))
        distances = torch.cdist(z_q, self.quantizer.codebook, p=2)
        soft_logits = -distances
        soft_logits = soft_logits.reshape(original_shape[:-1] + (self.quantizer.codebook_size,))
        if logits_type == "soft":
            return soft_logits
        elif logits_type == "hard":
            with torch.no_grad():
                indices = soft_logits.argmax(dim=-1)
                hard_logits = F.one_hot(indices, num_classes=self.quantizer.codebook_size).float().mul(100)
            return hard_logits + (soft_logits - soft_logits.detach())
    
    def decode(self, z_q):
        """Decode binary latent codes to images"""
        # z_q: (B, T, num_tokens, embed_dim) or (B, num_tokens, embed_dim)
        z_q = z_q.clone()
        shape = z_q.shape

        z_q = z_q.reshape((-1,) + shape[-2:]) # (N, num_tokens, embed_dim)
        z_q = z_q.reshape((z_q.shape[0], -1)) # (N, num_tokens * embed_dim)
        z_q = self.project_out(z_q) # (N, latent_dim * H' * W')
        z_q = z_q.reshape(shape[:-2] + self.shape[1:]) # (B, T, latent_dim, H', W') or (B, latent_dim, H', W')
        x_recon = self.decoder(z_q) # (B, T, C, H, W) or (B, C, H, W)

        return x_recon
    
    def forward(self, x):
        """Complete forward pass: encode -> quantize -> decode"""
        encode_result = self.encode(x)
        x_recon = self.decode(encode_result['z_q'])
        
        return {
            'x_recon': x_recon,
            'z_q': encode_result['z_q'],
        }