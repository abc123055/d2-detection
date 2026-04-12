import math
import json
import os

import torch
import torch.nn as nn


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    return x * cos + rotate_half(x) * sin


def compute_rope_2d(h, w, head_dim, theta=100.0, rescale=2.0, device=None, dtype=None):
    half_dim = head_dim // 2
    freq_dim = half_dim // 2

    freqs = 1.0 / (theta ** (torch.arange(0, freq_dim, device=device).float() / freq_dim))

    y_pos = torch.arange(h, device=device).float() * rescale
    x_pos = torch.arange(w, device=device).float() * rescale

    gy, gx = torch.meshgrid(y_pos, x_pos, indexing='ij')
    gy = gy.reshape(-1)
    gx = gx.reshape(-1)

    emb_x = gx.unsqueeze(-1) * freqs.unsqueeze(0)  # [h*w, freq_dim]
    emb_y = gy.unsqueeze(-1) * freqs.unsqueeze(0)  # [h*w, freq_dim]

    cos_x = torch.cat([emb_x.cos(), emb_x.cos()], dim=-1)  # [h*w, half_dim]
    sin_x = torch.cat([emb_x.sin(), emb_x.sin()], dim=-1)
    cos_y = torch.cat([emb_y.cos(), emb_y.cos()], dim=-1)
    sin_y = torch.cat([emb_y.sin(), emb_y.sin()], dim=-1)

    cos = torch.cat([cos_x, cos_y], dim=-1)  # [h*w, head_dim]
    sin = torch.cat([sin_x, sin_y], dim=-1)

    if dtype is not None:
        cos = cos.to(dtype)
        sin = sin.to(dtype)
    return cos, sin


class DINOv3Attention(nn.Module):
    def __init__(self, dim, num_heads=6, query_bias=True, key_bias=False,
                 value_bias=True, proj_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=query_bias)
        self.k_proj = nn.Linear(dim, dim, bias=key_bias)
        self.v_proj = nn.Linear(dim, dim, bias=value_bias)
        self.o_proj = nn.Linear(dim, dim, bias=proj_bias)

        self._rope_cos = None
        self._rope_sin = None
        self._num_prefix_tokens = 0

    def forward(self, x):
        B, N, C = x.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self._rope_cos is not None:
            prefix = self._num_prefix_tokens
            cos = self._rope_cos.unsqueeze(0).unsqueeze(0)  # [1, 1, h*w, head_dim]
            sin = self._rope_sin.unsqueeze(0).unsqueeze(0)

            q_patch = apply_rotary_pos_emb(q[:, :, prefix:, :], cos, sin)
            k_patch = apply_rotary_pos_emb(k[:, :, prefix:, :], cos, sin)

            q = torch.cat([q[:, :, :prefix, :], q_patch], dim=2)
            k = torch.cat([k[:, :, :prefix, :], k_patch], dim=2)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.o_proj(x)
        return x, attn


class DINOv3LayerScale(nn.Module):
    def __init__(self, dim, init_values=1.0):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        return x * self.lambda1


class DINOv3MLP(nn.Module):
    def __init__(self, dim, hidden_dim, bias=True):
        super().__init__()
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.act = nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


class DINOv3Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, query_bias=True, key_bias=False,
                 value_bias=True, proj_bias=True, mlp_bias=True, init_values=1.0,
                 layer_norm_eps=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attention = DINOv3Attention(dim, num_heads, query_bias, key_bias, value_bias, proj_bias)
        self.layer_scale1 = DINOv3LayerScale(dim, init_values)

        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.mlp = DINOv3MLP(dim, int(dim * mlp_ratio), bias=mlp_bias)
        self.layer_scale2 = DINOv3LayerScale(dim, init_values)

    def forward(self, x):
        y, attn = self.attention(self.norm1(x))
        x = x + self.layer_scale1(y)
        x = x + self.layer_scale2(self.mlp(self.norm2(x)))
        return x


class DINOv3Embeddings(nn.Module):
    def __init__(self, embed_dim=384, patch_size=16, num_register_tokens=4):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embeddings = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))


class DINOv3VisionTransformer(nn.Module):
    def __init__(self, patch_size=16, embed_dim=384, depth=12, num_heads=6,
                 mlp_ratio=4.0, num_register_tokens=4, rope_theta=100.0,
                 pos_embed_rescale=2.0, query_bias=True, key_bias=False,
                 value_bias=True, proj_bias=True, mlp_bias=True,
                 layerscale_value=1.0, layer_norm_eps=1e-5):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens
        self.rope_theta = rope_theta
        self.pos_embed_rescale = pos_embed_rescale

        self.embeddings = DINOv3Embeddings(embed_dim, patch_size, num_register_tokens)

        self.layer = nn.ModuleList([
            DINOv3Block(embed_dim, num_heads, mlp_ratio, query_bias, key_bias,
                        value_bias, proj_bias, mlp_bias, layerscale_value, layer_norm_eps)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

    @property
    def blocks(self):
        return self.layer

    def prepare_tokens(self, x):
        B, C, H, W = x.shape

        x = self.embeddings.patch_embeddings(x)  # [B, D, H/P, W/P]
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, h*w, D]

        cls = self.embeddings.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if self.num_register_tokens > 0:
            reg = self.embeddings.register_tokens.expand(B, -1, -1)
            x = torch.cat([x[:, :1], reg, x[:, 1:]], dim=1)

        head_dim = self.embed_dim // self.num_heads
        cos, sin = compute_rope_2d(h, w, head_dim, self.rope_theta,
                                   self.pos_embed_rescale, x.device, x.dtype)

        num_prefix = 1 + self.num_register_tokens
        for blk in self.layer:
            blk.attention._rope_cos = cos
            blk.attention._rope_sin = sin
            blk.attention._num_prefix_tokens = num_prefix

        return x

    @classmethod
    def from_pretrained(cls, model_dir):
        from safetensors.torch import load_file

        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path) as f:
            config = json.load(f)

        model = cls(
            patch_size=config['patch_size'],
            embed_dim=config['hidden_size'],
            depth=config['num_hidden_layers'],
            num_heads=config['num_attention_heads'],
            mlp_ratio=config['intermediate_size'] / config['hidden_size'],
            num_register_tokens=config.get('num_register_tokens', 4),
            rope_theta=config.get('rope_theta', 100.0),
            pos_embed_rescale=config.get('pos_embed_rescale', 2.0),
            query_bias=config.get('query_bias', True),
            key_bias=config.get('key_bias', False),
            value_bias=config.get('value_bias', True),
            proj_bias=config.get('proj_bias', True),
            mlp_bias=config.get('mlp_bias', True),
            layerscale_value=config.get('layerscale_value', 1.0),
            layer_norm_eps=config.get('layer_norm_eps', 1e-5),
        )

        weights_path = os.path.join(model_dir, 'model.safetensors')
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=True)

        return model
