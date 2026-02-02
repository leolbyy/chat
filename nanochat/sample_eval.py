"""
This file only serve as a placeholder and holds very simple funtions.
Should be implemented with batched inference in future
"""


def get_response(model, tokenizer, prompt, max_tokens, temperature=1.0, top_k=None):
    assistant_end = tokenizer.encode_special('<|assistant_end|>')
    bos = tokenizer.encode_special('<|bos|>')

    tokens = tokenizer.encode(prompt, prepend=bos)
    output = []
    for next_token in model.generate(tokens, max_tokens, temperature=temperature, top_k=top_k):
        if (next_token == assistant_end or next_token == bos) or len(output) >= max_tokens:
            break
        else:
            output.append(next_token)
    output_str = tokenizer.decode(output)
    return output_str