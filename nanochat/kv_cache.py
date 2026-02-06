import torch
"""
In this approch, we apply hard restict on seq_len <= max_seq_len, i.e. no extending on RoPE.
This is causes some problem. For example, even with sufficient history tokens, the first input token will have less than max_seq_len tokens to pay attention to.

A more elegant way i can think of is using sliding windows. KV cache will have seq_len = 2 * max_seq_len, so that we can make sure all tokens can have enough tokens to pay attention to.
However, as reported in original nanochat project, naive implementation of sliding window attention (without using flash attention kernel) will be awfully slow.
Plus, I have a very small GPU VRAM. 
Therefore, I will choose the first non-elegant, easy, fast & memory-saving approch.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
This implementation DOES NOT work Flash Attention Kernel!!!!!!!!!!
My GPU does not support FA3 so it works for me. 
DO NOT USE THIS KV CACHE IMPLEMENTATION IF YOUR GPU IS HOPEER OR NEWER!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

class KVCache:
    def __init__(self, batch_size, seq_len, num_heads, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim

        self.k_cache = torch.randn(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.randn(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.token_pos = [0] * batch_size
    
    def get_pos(self):
        return self.cache_seqlens.tolist()
    
    def get_layer_cache(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]
    
    def update_layer_cache(self, layer_idx, k, v, query_lens):
        k_cache, v_cache = self.get_layer_cache(layer_idx)
        cache_seqlens = self.get_pos()
        for i in range(self.batch_size):
            new_seq_len = query_lens[i] if query_lens is not None else 1
            current_len = cache_seqlens[i]
            remaining = self.max_seq_len - current_len
            if remaining >= new_seq_len:
                k_cache[i, current_len:current_len + new_seq_len, :, :] = k[i, :new_seq_len, :, :]
                v_cache[i, current_len:current_len + new_seq_len, :, :] = v[i, :new_seq_len, :, :]
            else:
                offset = remaining - new_seq_len
                start_pos = self.max_seq_len - new_seq_len
                k_cache[i] = k_cache[i].roll(offset, dims=0)
                v_cache[i] = v_cache[i].roll(offset, dims=0)
                k_cache[i, start_pos:, :, :] = k[i, :new_seq_len, :, :] # we can safely insert to start_pos to end since we only clean new_seq_len to fit
                v_cache[i, start_pos:, :, :] = v[i, :new_seq_len, :, :]
    
    def update_pos(self, query_lens):
        for i in range(self.batch_size):
            if query_lens is not None:
                self.cache_seqlens[i] = min(self.cache_seqlens[i] + query_lens[i], self.max_seq_len)
                self.token_pos[i] = self.token_pos[i] + query_lens[i] 
            else:
                self.cache_seqlens[i] = min(self.cache_seqlens[i] + 1, self.max_seq_len)
                self.token_pos[i] = self.token_pos[i] + 1