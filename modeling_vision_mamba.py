import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum

from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling, ImageClassifierOutputWithNoAttention
from configuration_vision_mamba import VisionMambaConfig

class VisionMambaPreTrainedModel(PreTrainedModel):
    config_class = VisionMambaConfig
    base_model_prefix = "vision_mamba"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

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

class VisionMambaModel(VisionMambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            stride=config.stride,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim
        )

        if config.if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        if config.if_abs_pos_embed:
            num_patches = self.patch_embed.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + (1 if config.if_cls_token else 0), config.embed_dim))

        self.blocks = nn.ModuleList([MambaBlock(config, i) for i in range(config.depth)])

        self.norm = RMSNorm(config.embed_dim) if config.use_final_norm else nn.Identity()

        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        x = self.patch_embed(pixel_values)

        if self.config.if_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            if self.config.use_middle_cls_token:
                x = torch.cat((x[:, :x.size(1)//2], cls_token, x[:, x.size(1)//2:]), dim=1)
            else:
                x = torch.cat((cls_token, x), dim=1)

        if self.config.if_abs_pos_embed:
            x = x + self.pos_embed

        hidden_states = []
        for block in self.blocks:
            x = block(x)
            if output_hidden_states:
                hidden_states.append(x)

        x = self.norm(x)

        if self.config.final_pool_type == 'mean':
            pooled_output = x.mean(dim=1)
        elif self.config.if_cls_token:
            pooled_output = x[:, 0]
        else:
            pooled_output = x[:, -1]

        if not return_dict:
            return (x, pooled_output) + (hidden_states,) if output_hidden_states else (x, pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=x,
            pooler_output=pooled_output,
            hidden_states=hidden_states if output_hidden_states else None,
        )

class VisionMambaForImageClassification(VisionMambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_classes
        self.vision_mamba = VisionMambaModel(config)
        self.classifier = nn.Linear(config.embed_dim, config.num_classes) if config.num_classes > 0 else nn.Identity()

        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vision_mamba(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )