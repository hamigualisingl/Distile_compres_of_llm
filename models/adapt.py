from torch.nn.init import trunc_normal_
import numpy as np
import torch.utils.checkpoint
import torch
from torch import nn
import math
from torch.nn import functional as F

def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos

# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Prompt_Adapt(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            norm_layer=nn.LayerNorm,
            k_len = None
    ):
        super().__init__()
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float()
        ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None, key_padding_mask=None):

        pos_embed = get_abs_pos(self.pos_embed, x.size(1))

        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)

        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.unsqueeze(1),
            x + pos_embed.unsqueeze(1),
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]
        return out.permute(1, 0, 2)

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)

import torch.utils.checkpoint
from functools import partial
import torch
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
from torch import nn
import numpy as np

def sinusoids(max_length=3000, embed_dim=1280, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert embed_dim % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (embed_dim // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(embed_dim // 2))
    scaled_time = torch.arange(max_length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class Adaptor(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
            self,
            num_queries=None,
            input_dim=1024,
            output_dim=3584,
            input_len=1024,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(Adaptor, self).__init__()
        self.num_queries = num_queries
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_len = input_len
        self.num_heads = num_heads = self.output_dim // 128
        self.query = nn.Parameter(torch.zeros(self.num_queries, self.output_dim))
        trunc_normal_(self.query, std=.02)
        self.register_buffer("k_positional_embedding", sinusoids(self.input_len, self.output_dim))
        self.register_buffer("q_positional_embedding", sinusoids(self.num_queries, self.output_dim))

        if self.input_dim is not None and self.input_dim != self.output_dim:
            self.kv_proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()
        self.attn = nn.MultiheadAttention(self.output_dim, num_heads)
        self.ln_q = norm_layer(self.output_dim)
        self.ln_kv = norm_layer(self.output_dim)

        self.ln_post = norm_layer(self.output_dim)
        self.proj = nn.Parameter((self.output_dim ** -0.5) * torch.randn(self.output_dim, self.output_dim))
        self.apply(self._init_weights)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
               input shape:[BS,seq_len,input_dim]
               out shape:[BS,query_len,out_dim]
        """
        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(
            self._repeat(q, N) + self.q_positional_embedding.unsqueeze(1),
            x + self.k_positional_embedding.unsqueeze(1),
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]
        out = out.permute(1, 0, 2)
        out = self.ln_post(out)
        out = out @ self.proj
        return out

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)
