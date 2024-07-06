import math
import random
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling, ImageClassifierOutputWithNoAttention

from configuration_vision_mamba import VisionMambaConfig
from rope import VisionRotaryEmbeddingFast

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
    
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class MambaBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.norm = RMSNorm(config.embed_dim)
        self.mixer = Mamba(config, layer_idx=layer_idx)

    def forward(self, hidden_states, inference_params=None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        hidden_states = residual + hidden_states
        return hidden_states

class Mamba(nn.Module):
    def __init__(self, config: VisionMambaConfig, layer_idx=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

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
            kernel_size=config.d_conv,
            bias=config.conv_bias,
            groups=self.d_inner,
            padding=config.d_conv - 1,
        )

        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj
        dt_init_std = self.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt_proj bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)

    def forward(self, x, inference_params=None):
        B, L, _ = x.shape

        xz = rearrange(self.in_proj(x), "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)

        x = self.act(self.conv1d(x)[..., :L])

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=L)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()

        A = -torch.exp(self.A_log.float())

        y = self.selective_scan(x, dt, A, B, C, self.D.float(), z)
        y = rearrange(y, "b d l -> b l d")
        
        out = self.out_proj(y)
        return out

    def selective_scan(self, u, delta, A, B, C, D, z):
        deltaA = torch.exp(torch.einsum('blr,dn->bldrn', F.softplus(delta), A))
        deltaB_u = torch.einsum('blr,brl,blr->bldr', delta, B, u)
        
        x = torch.zeros(u.shape[0], u.shape[2], A.shape[1], device=u.device, dtype=u.dtype)
        ys = []
        for l in range(u.shape[1]):
            x = torch.einsum('bdn,bdrn->bdr', x, deltaA[:, l]) + deltaB_u[:, l]
            y = torch.einsum('bdr,br->bd', x, C[:, :, l])
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + u * D[None, None, :]
        return y * self.act(z)

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
            if config.use_middle_cls_token:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            else:
                self.cls_token_head = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        if config.if_abs_pos_embed:
            num_patches = self.patch_embed.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + (1 if config.if_cls_token else 0), config.embed_dim))
            self.pos_drop = nn.Dropout(p=config.drop_rate)

        if config.if_rope:
            self.rope = VisionRotaryEmbeddingFast(
                dim=config.embed_dim // 2,
                pt_seq_len=config.pt_hw_seq_len,
                ft_seq_len=self.patch_embed.num_patches
            )

        self.blocks = nn.ModuleList([MambaBlock(config, i) for i in range(config.depth)])

        self.norm = RMSNorm(config.embed_dim) if config.use_final_norm else nn.Identity()

        self.if_cls_token = config.if_cls_token
        self.use_middle_cls_token = config.use_middle_cls_token
        self.if_abs_pos_embed = config.if_abs_pos_embed
        self.if_rope = config.if_rope
        self.if_rope_residual = config.if_rope_residual
        self.if_bidirectional = config.bimamba_type != "none"
        self.flip_img_sequences_ratio = config.flip_img_sequences_ratio
        self.final_pool_type = config.final_pool_type

        self.post_init()

    def forward_features(self, x, inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False):
        x = self.patch_embed(x)
        B, M, _ = x.shape

        if self.if_cls_token:
            if self.use_middle_cls_token:
                cls_token = self.cls_token.expand(B, -1, -1)
                token_position = M // 2
                x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
            else:
                cls_token_head = self.cls_token_head.expand(B, -1, -1)
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)
                token_position = [0, M + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
            M = x.shape[1]

        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        if if_random_token_rank:
            shuffle_indices = torch.randperm(M)
            x = x[:, shuffle_indices, :]
            if isinstance(token_position, list):
                token_position = [torch.where(shuffle_indices == pos)[0].item() for pos in token_position]
            else:
                token_position = torch.where(shuffle_indices == token_position)[0].item()

        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio > 0 and (self.flip_img_sequences_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True

        if not self.if_bidirectional:
            for layer in self.blocks:
                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])

                if self.if_rope:
                    hidden_states = self.rope(hidden_states)

                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])

                hidden_states = layer(hidden_states, inference_params=inference_params)
        else:
            # Bidirectional processing
            for i in range(0, len(self.blocks), 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)

                # Forward direction
                hidden_states_f = self.blocks[i](hidden_states, inference_params=inference_params)
                
                # Backward direction
                hidden_states_b = self.blocks[i+1](hidden_states.flip([1]), inference_params=inference_params)
                hidden_states_b = hidden_states_b.flip([1])

                # Combine forward and backward
                hidden_states = hidden_states_f + hidden_states_b

        hidden_states = self.norm(hidden_states)

        if self.if_cls_token:
            if self.use_middle_cls_token:
                return hidden_states[:, token_position, :]
            else:
                return (hidden_states[:, token_position[0], :] + hidden_states[:, token_position[1], :]) / 2

        if self.final_pool_type == 'none':
            return hidden_states[:, -1, :]
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            return hidden_states.max(dim=1)[0]
        elif self.final_pool_type == 'all':
            return hidden_states
        else:
            raise NotImplementedError

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        if_random_cls_token_position: bool = False,
        if_random_token_rank: bool = False,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.forward_features(
            pixel_values, 
            if_random_cls_token_position=if_random_cls_token_position, 
            if_random_token_rank=if_random_token_rank
        )

        if not return_dict:
            return (hidden_states,)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=hidden_states,
            hidden_states=None,
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

        pooled_output = outputs.pooler_output if return_dict else outputs[0]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )