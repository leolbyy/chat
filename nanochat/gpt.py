import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn
import torch.nn.functional as F







@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304 
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768

# Just implement here. Should not be used as it not as efficient as the pytorch native implementation.
class RMSNorm_Old_School(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keep_dim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multi-head attetion, [bs, seq_len, n_head, head_dim]
    d = x.shape[3] // 2

    x1, x2 = x[...,:d], x[...,d:] # Not the original implementation in paper but mathmatically equivalent
    y1 = x1 * cos + x2 * sin 
    y2 = x1 * (-sin) + x2 * cos

    return torch.cat([y1, y2], 3) # [bs, seq_len, n_head, head_dim // 2] --> [bs, seq_len, n_head, head_dim]


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.embd // self.n_head

        assert self.n_embd % self.n_head == 0 # Make sure sum of heads equal to original dim
        assert self.n_kv_head <= self.n_head
        assert self.n_head % self.n_kv_head == 0

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)

        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    
    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size() # Keep same notation as nanochat project for easy development. [bs, seq_len, n_head * head_dim]

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply rotary embedding to q and k
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # [bs, n_head, seq_len, head_dim]


        if kv_cache is not None: # We are now in inference
            k, v = kv_cache.insert_kv(self.layer_idx, k, v) # TODO check kv_cache implementation
        
        # 
        Tq = q.size(2)
        Tk = k.size(2)

        enable_gqa = self.n_kv_head != self.n_head

        if kv_cache is None or Tq == Tk:
            # kv_cache is None --> Training. KV cache is disabled
            # Tq == Tk: Means this is the very first query. KV cache is empty for now
            y = F/scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
