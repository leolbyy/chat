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


def flash_attn_func(q, k, v, causal=True, window_size=(-1, 0)):
    if _fa3 is not None:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    

    assert q.size(1) == k.size(1) == v.size(1)
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)

    seq_len = q.size(2)
    device = q.device
    window = window_size[0]

    if window < 0 or window >= seq_len:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
    if seq_len == 1:
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = mask & ((row_idx - col_idx) <= window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)
    

def flash_attn_with_kvcache(q, k_cache, v_cache, k, v, cache_seqlens, causal=False, window_size=(-1, 0)):
    if _fa3 is not None:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )
    
    # k_cache, v_cache is pre-allocated tensor, (bs, seq_len_max, n_kv, head_dim)
    assert k_cache.shape == v_cache.shape
    assert q.size(1) == k.size(1) == v.size(1)
    
    q, k_cache, v_cache, k, v, = q.transpose(1, 2), k_cache.transpose(1, 2), v_cache.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    pos = cache_seqlens[0].item()
    
    seq_len = q.size(2)
    device = q.device
    enable_gqa = q.size(1) != k.size(1)

    k_cache[:, :, pos: pos + seq_len, :] = k
    v_cache[:, :, pos: pos + seq_len, :] = v

    k_full = k_cache[:, :, :pos + seq_len, :]
    v_full = v_cache[:, :, :pos + seq_len, :]

    mask_cache = torch.ones(seq_len, pos, device=device, dtype=torch.bool)
    mask_new = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    mask = torch.cat([mask_cache, mask_new], dim=2)

    if window < -1:
        mask = mask
    else:
        mask = mask.triu(diagonal=(pos - window))

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

        

# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
