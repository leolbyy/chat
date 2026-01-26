import torch


class KVCache:
    def __init__(self, batch_size, seq_len, num_heads, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim

        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    def reset(self):
        self.cache_seqlens.zero_()
    
    def get_pos(self):
        return self.cache_seqlens.tolist()
    
    def get_layer_cache(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]
    
    def update_seqlens(self, num_tokens):
        assert num_tokens.ndim == 1
        assert num_tokens.shape == self.cache_seqlens.shape
        self.cache_seqlens += num_tokens
    
    def update(self, k, v):
        
