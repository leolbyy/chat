import math
import torch
import torch.nn.functional as F

_fa3 = None

def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper+ GPU)."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major < 9:  # Hopper is sm90
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None

_fa3 = _load_flash_attention_3()


def flash_attn_func(q, k, v, causal=True):
    if _fa3 is not None:
        return _fa3.flash_attn_func(q, k, v, causal=causal)
    
    assert q.size(1) == k.size(1) == v.size(1)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)

    seq_len = q.size(2)
        
    if seq_len == 1:
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)


def flash_attn_with_kvcache(q, k, v, cache_seqlens):
    # Remove FA3 support becuase current kvcache implementation does not support FA3.
    # This is acceptable to be becuase my serving GPU does not support FA3 anyway.
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # This attention will be awfully slow since I implement it from scratch with python.
    # But it should be acceptable since this function is only used during inference.abs
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Expect input shape: [bs, seq_len, n_head, head_dim]
    assert k.shape == v.shape
    assert q.shape[1] <= k.shape[1]
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    
    attention_mask = torch.ones(q.shape[0], q.shape[2], k.shape[2], dtype=torch.bool, device=q.device) # [batch_size, query_seq_len, kv_seq_len]
    attention_bias = torch.zeros(q.shape[0], q.shape[2], k.shape[2], dtype=q.dtype, device=q.device)
    for i in range(q.shape[0]):
        attention_mask[i] = attention_mask[1].tril(diagonal=cache_seqlens[i])
        attention_bias.masked_fill_(~attention_mask[i], -float('inf'))    
    attention_bias = attention_bias.unsqueeze(1)

    scale_factor = 1 / math.sqrt(q.size(-1))

    if q.size(1) != k.size(1): # enable gqa
        n_rep = q.size(1) // k.size(1)
        bs, n_kv_head, slen, head_dim = k.shape
        k = k[:, :, None, :, :].expand(bs, n_kv_head, n_rep, slen, head_dim).reshape(bs, n_kv_head * n_rep, slen, head_dim)
        v = v[:, :, None, :, :].expand(bs, n_kv_head, n_rep, slen, head_dim).reshape(bs, n_kv_head * n_rep, slen, head_dim)

    attention_score = q @ k.transpose(-2, -1) * scale_factor
    attention_score = attention_score + attention_bias
    attention_score = torch.softmax(attention_score, dim=-1)

    return attention_score @ v



# def flash_attn_with_kvcache(q, k_cache, v_cache, k, v, query_lens, cache_seqlens, causal=False):
#     if _fa3 is not None:
#         return _fa3.flash_attn_with_kvcache(
#             q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
#             causal=causal, window_size=window_size
#         )
    
#     # k_cache, v_cache is pre-allocated tensor, (bs, seq_len_max, n_kv, head_dim)
#     assert k_cache.shape == v_cache.shape
#     assert q.size(1) == k.size(1) == v.size(1)
    
#     q, k_cache, v_cache, k, v, = q.transpose(1, 2), k_cache.transpose(1, 2), v_cache.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
#     pos = cache_seqlens[0].item()
    
#     seq_len = q.size(2)
#     device = q.device
#     enable_gqa = q.size(1) != k.size(1)

#     k_cache[:, :, pos: pos + seq_len, :] = k
#     v_cache[:, :, pos: pos + seq_len, :] = v

#     k_full = k_cache[:, :, :pos + seq_len, :]
#     v_full = v_cache[:, :, :pos + seq_len, :]

#     mask_cache = torch.ones(seq_len, pos, device=device, dtype=torch.bool)
#     mask_new = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
#     mask = torch.cat([mask_cache, mask_new], dim=2)

#     if window < -1:
#         mask = mask
#     else:
#         mask = mask.triu(diagonal=(pos - window))

#     return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

        

# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
