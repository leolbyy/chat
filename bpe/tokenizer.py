"""
This script is used to train / load a tokenizer, and then fit into a tiktoken tokenizer.
This is to speed up the encode and decode process for a faster model training.
"""

import os
import tiktoken
import pickle
from functools import lru_cache
from bpe.tokTrainer import TokenizerTrainer


PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    # "<|python_start|>", # assistant invokes python REPL tool
    # "<|python_end|>",
    # "<|output_start|>", # python REPL outputs back to assistant
    # "<|output_end|>",
]

class BaseTokenizer:
    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, vocab_size, max_char, n_parallel=8):
        trainer = TokenizerTrainer(vocab_size, max_char, PATTERN, SPECIAL_TOKENS, n_parallel=n_parallel)
        mergeable_ranks, special_tokens, pattern = trainer.train_from_iterator()
        enc = tiktoken.Encoding(
            name='pybpe',
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens
        )
        return cls(enc, '<|bos|>')

    @classmethod
    def load_from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, 'tokenizer.pkl')
        with open(pickle_path, 'rb') as f:
            mergeable_ranks, special_tokens, pattern = pickle.load(f)
        enc = tiktoken.Encoding(
            name='pybpe',
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens
        )
        return cls(enc, '<|bos|>')


    def get_vocab_size(self):
        return self.enc.n_vocab

    
    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])
    
    @lru_cache(maxsize=32)
    def encode_special(self, text):
        assert text in self.enc.special_tokens_set, f"{text} is not a special token!"
        return self.enc.encode_single_token(text)
    
    def get_bos_token_id(self):
        return self.bos_token_id
    
    def encode(self, text, prepend=None, append=None, num_threads=8):
        # Need more checks on prepend and append
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids
    

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    
    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, 'tokenizer.pkl')
        data = (self.enc._mergeable_ranks, self.enc._special_tokens, PATTERN)
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved tokenizer encoding to {pickle_path}")
    
    def render_conversation(self, messages):
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        bos = self.encode_special('<|bos|>')
        conversation = [bos]
        # some conversation may contain system prompt
        if messages[0]['role'] == 'system':
            messages[1]['content'] = messages[0]['content'] + messages[1]['content']
            messages = messages[1:]
        messages_content = []
        for i, message in enumerate(messages):
            if i % 2 == 0:
                assert message['role'] == 'user', f"{message['role']}"
            else:
                assert message['role'] == 'assistant'
            if message['role'] == 'user':
                conversation.append(user_start)
                conversation += self.encode(message['content'])
                conversation.append(user_end)
            else:
                conversation.append(assistant_start)
                conversation += self.encode(message['content'])
                conversation.append(assistant_end)
        return conversation, None

    def render_conversation_batch(self, messages_list):
        conversation_list = []
        for messages in messages_list:
            conversation_list.append(self.render_conversation(messages))
        return conversation_list
        


def get_tokenizer(tokenizer_dir):
    return BaseTokenizer.load_from_directory(tokenizer_dir)

def get_token_bytes(tokenizer_dir, device="cpu"):
    import torch
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes