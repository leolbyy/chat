import os
import math
from functools import partial
from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn
import torch.nn
import torch.nn.functional as F

from utils.common import get_dist_info

from nanochat.flash_attention import flash_attn
from nanochat.adamw import DistAdamW
from nanochat.muon import Muon, DistMuon


@dataclass
class GPTConfig:
    seq_len: int = 1024
    vocab_size: int = 50304 
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768

# Just implement here. Should not be used as it not as efficient as the pytorch native implementation.
class MyRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keep_dim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def my_attention(q, k, v, enable_gqa=False):
    # q, k, v should be in [bs, n_head, seq_len, head_dim]
    assert k.size == v.size
    assert q.shape(-2) <= k.shape(-2)
    
    attention_mask = torch.ones(q.shape(-2), k.shape(-2), dtype=q.dtype, device=q.device).triu(k.shape(-2) - q.shape(-2) + 1)
    attention_mask = attention_mask.masked_fill(attention_mask == 1, -float('inf'))

    scale_factor = 1 / math.sqrt(q.size(-1))

    if q.size(2) != k.size(2): # enable gqa
        n_rep = q.size(2) // k.size(2)
        bs, n_kv_head, slen, head_dim = k.shape()
        k = k[:, :, None, :, :].expand(bs, n_kv_head, n_rep, slen, head_dim).reshape(bs, n_kv_head * n_rep, slen, head_dim)
        v = v[:, :, None, :, :].expand(bs, n_kv_head, n_rep, slen, head_dim).reshape(bs, n_kv_head * n_rep, slen, head_dim)

    attention_score = q @ k * scale_factor
    attention_score = attention_score + attention_mask
    attention_score = torch.softmax(attetnion_score, dim=-1)

    return attention_score @ v



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
        self.head_dim = self.n_embd // self.n_head

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


        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    # decoder block
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
    
    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))

        return x
    
class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config

        # pad vocab size to be divisible by world_size.
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            pass # TODO logging
        
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(padded_vocab_size, config.n_embd),
            'h': nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)])
        })

        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)

        # per-layer learnable scalars
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Seperate parameters so they can have different optimizer treatment
        # TODO review this part
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))


        self.rotary_seq_len = config.seq_len * 10
        self.max_seq_len = self.rotary_seq_len # TODO
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = (3 ** 0.5) * (n_embd ** -0.5)
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # init projection weights to zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.0)
        
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # if self.transformer.wte.weight.device.type == 'cuda':
        #     self.transformer.wte.to(dtype=torch.bfloat16)
        self.transformer.wte.to(dtype=torch.bfloat16)

    
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))

        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # [bs, seq_len, n_head, head_dim]
        return cos, sin



    def get_device(self):
        return self.transformer.wte.weight.device

    
    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, the term 12 * l * h * q * t accounts for key @ query matmul flops inside attention.
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.seq_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t # need seq_len becuase self attention need to calculate with all tokens in seq
        return num_flops_per_token
    
    def num_scaling_params(self):
        nparams = sum(p.numel() for p in self.parameters())
        return nparams
    
    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(resid_params) + len(x0_params)
    
        dmodel_lr_scale = (model_dim / 768) ** -0.5

        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0)
        adamw_groups = [
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=resid_params, lr=scalar_lr * 0.01), # resid is very sensitive, so use a small lr
            dict(params=x0_params, lr=scalar_lr)
        ]
        adamw_optimizer = AdamWFactory(adamw_groups, **adamw_kwargs)

        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)

        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group['initial_lr'] = group['lr']
        return optimizers


    def forward(self, idx, targets=None, kv_cache=None, seqlens=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of the shape (1, seq_len, 1, head_dim // 2))
        assert T <= self.cos.size(1), f'Seqeunce length grew beyond the maximum rotary embeddings cache: {T} > {self.cos.size(1)}'
        assert idx.device == self.cos.device, f'Rotaty embeddings cache and idx are on different devices: idx --> {idx.device}, RoPE --> {self.cos.device}'
        assert self.cos.dtype == torch.bfloat16, 'Rotary embedding must be in bfloat16 format'

        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = (self.cos[:, T0:T0 + T, :, :], self.sin[:, T0:T0 + T, :, :])

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, layer in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = layer(x, cos_sin, kv_cache)
        x = norm(x)
        
        softcap = 15
        logits = self.lm_head(x) # [bs, seq_len, vocab_size] 
        logits = logits[..., :self.config.vocab_size] # remove padding
        assert logits.size() == (B, T, self.config.vocab_size), 'logits shape error'
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap) # Squash with smooth. Talyor Expansion Proof

        if targets is not None:
            # Training
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            return logits
        
    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        # TODO Implement my own version of batched inference, with KV Cache support
        assert isinstance(tokens, list)

        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1:, :]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
