"""
This file only serve as a placeholder and holds very simple funtions.
Should be implemented with batched inference in future
"""
from nanochat.kv_cache import KVCache


def get_response(model, tokenizer, prompt, max_tokens, temperature=1.0, top_k=None):
    assert isinstance(prompt, str)
    assistant_end = tokenizer.encode_special('<|assistant_end|>')
    bos = tokenizer.encode_special('<|bos|>')

    tokens = [tokenizer.encode(prompt, prepend=bos)]
    output = []
    for next_token in model.generate(tokens, max_tokens, temperature=temperature, top_k=top_k):
        next_token = next_token[0][0]
        if (next_token == assistant_end or next_token == bos):
            break
        elif len(output) + 1 >= max_tokens:
            output.append(next_token)
            break
        else:
            output.append(next_token)
    output_str = tokenizer.decode(output)
    return output_str

def get_response_batch_kvcache(model, tokenizer, prompts, max_tokens, kv_cache=None, temperature=1.0, top_k=None):
    assert isinstance(prompts, list)
    assert isinstance(prompts[0], str)

    assistant_end = tokenizer.encode_special('<|assistant_end|>')
    bos = tokenizer.encode_special('<|bos|>')

    tokens = tokenizer.encode(prompts, prepend=bos)
    query_lens = [len(token) for token in tokens]
    
    # pad to same length
    max_len = max(query_lens)
    for token in tokens:
        if len(token) < max_len:
            token.extend([bos] * (max_len - len(token)))

    outputs = [[] for _ in range(len(prompts))]
    completed = [False] * len(prompts)
    for next_tokens in model.generate(tokens, max_tokens, kv_cache, query_lens, temperature=temperature, top_k=top_k):
        for i, next_token in enumerate(next_tokens):
            if completed[i]:
                continue
            next_token = next_token[0]
            if (next_token == assistant_end or next_token == bos):
                completed[i] = True
                continue
            else:
                outputs[i].append(next_token)
        if len(outputs[0]) == max_tokens:
            break
        elif sum(completed) == len(prompts):
            break
        else:
            continue
    output_strs = [tokenizer.decode(output) for output in outputs]
    return output_strs