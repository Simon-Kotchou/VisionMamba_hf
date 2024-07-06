import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling, ImageClassifierOutputWithNoAttention

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = ((img_size - patch_size) // stride + 1, (img_size - patch_size) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class MambaBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.norm = RMSNorm(config.embed_dim)
        self.mixer = Mamba(config)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Mamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.d_model = config.embed_dim
        self.d_state = config.d_state
        self.d_conv = config.d_conv
        self.expand = config.expand_factor
        self.d_inner = int(self.expand * self.d_model)
        
        self.dt_rank = math.ceil(self.d_model / 16) if config.dt_rank == "auto" else config.dt_rank
        
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=config.bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=config.kernel_size,
            bias=config.conv_bias,
            groups=self.d_inner,
            padding=config.kernel_size - 1,
        )
        
        self.act = nn.SiLU()
        
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        self.A = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)
        
    def forward(self, x):
        B, L, _ = x.shape
        
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.chunk(2, dim=-1)
        
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :L]
        x = rearrange(x, 'b d l -> b l d')
        
        x = self.act(x)
        
        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        delta = F.softplus(self.dt_proj(delta))
        
        y = self.selective_scan(x, delta, A=self.A, B=B, C=C, D=self.D)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)
        
        return output
    
    def selective_scan(self, u, delta, A, B, C, D):
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in d_state -> b l d_in d_state'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l d_state, b l d_in -> b d_in l d_state')
        
        x = torch.zeros((u.shape[0], self.d_inner, self.d_state), device=u.device)
        ys = []
        for i in range(u.shape[1]):
            x = deltaA[:, i] * x + deltaB_u[:, :, i]
            y = einsum(x, C[:, i], 'b d_in d_state, b d_state -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        
        y = y + u * D
        
        return y