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
    def __init__(self, config: VisionMambaConfig):
        super().__init__()
        self.config = config

        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=config.bias)

        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        self.act = nn.SiLU()

        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + config.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # Initialize dt_proj
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt_proj bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(torch.arange(1, config.d_state + 1), 'n -> d n', d=config.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x):
        B, L, _ = x.shape
        x_copy = x 

        x = self.norm(x)
        x_and_res = self.in_proj(x)
        x, res = x_and_res.chunk(2, dim=-1)

        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[:, :, :L]
        x = rearrange(x, 'b d l -> b l d')
        
        x = self.act(x)

        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.transpose(-1, -2)
        dt = rearrange(dt, 'd (b l) -> b l d', l=L)
        B = rearrange(B, '(b l) dstate -> b dstate l', l=L).contiguous()
        C = rearrange(C, '(b l) dstate -> b dstate l', l=L).contiguous()

        y = self.selective_scan(x, dt, -torch.exp(self.A_log.float()), B, C, self.D.float())

        y = y * F.silu(res)
        
        output = self.out_proj(y) + x_copy

        return output

    def selective_scan(self, u, delta, A, B, C, D):
        deltaA = torch.exp(torch.einsum('blr,dn->bldrn', F.softplus(delta + self.dt_proj.bias), A))
        deltaB_u = torch.einsum('blr,brl,blr->bldr', delta, B, u)
        
        x = torch.zeros(u.shape[0], u.shape[2], A.shape[1], device=u.device, dtype=u.dtype)
        ys = []
        for l in range(u.shape[1]):
            x = torch.einsum('bdn,bdrn->bdr', x, deltaA[:, l]) + deltaB_u[:, l]
            y = torch.einsum('bdr,br->bd', x, C[:, :, l])
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + u * D[None, None, :]
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