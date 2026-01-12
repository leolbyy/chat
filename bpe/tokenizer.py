import os
import regex as re
import unicodedata
import multiprocessing
from tqdm import tqdm
from itertools import chain
from collections import Counter


PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s


class BaseTokenizer():
    def __init__(self, pattern=None):
        self.pattern = pattern if pattern is not None else PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.vocab = {}
        self.mergeable_rank = {}
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.mergeable_rank.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    
    def _load_text(self, text_iterator, max_char):
        char_count = 0
        word_dict = Counter()
        for text in text_iterator:
            text = ''.join(text)
            if char_count < max_char:
                word_list = re.findall(self.compiled_pattern, text)
                word_dict.update(word_list)
                char_count += len(text)
            else:
                break
        return word_dict
    

    def _get_stats_chunk(self, ids_dict):
        stats_final = Counter() # {(int, int): int}
        for ids in ids_dict.keys():
            id_pair = zip(ids, ids[1:])
            stats = Counter(id_pair)
            stats = {k: v * ids_dict[ids] for k, v in stats.items()}
            stats_final.update(stats)
        return stats_final

    def _merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def _merge_dict(self, ids_dict, pair, idx):
        new_ids_dict = {}
        for ids in ids_dict.keys():
            new_ids = self._merge(ids, pair, idx)
            new_ids_dict[tuple(new_ids)] = ids_dict[ids]
        return new_ids_dict


    def _parallel_train(self, worker_id, ids_dict, task_queue, result_queue):
        print(f'Started {worker_id} process.')

        while True:
            command, payload = task_queue.get()
            if command == 'STOP':
                break
            elif command == 'MERGE':
                pair, idx = payload
                ids_dict = self._merge_dict(ids_dict, pair, idx)
            elif command == 'COUNT':
                stats = self._get_stats_chunk(ids_dict)
                result_queue.put(stats)
            else:
                raise ValueError(f'Command {command} not understood.')


    def train_from_text(self, text_iterator, vocab_size, max_char, n_parallel=None):
        assert vocab_size >= 256
        n_steps = vocab_size - 256

        word_dict = self._load_text(text_iterator, max_char=max_char)
        ids_dict = {tuple(word.encode('utf-8')): v for word, v in word_dict.items()}

        vocab = {idx: bytes([idx]) for idx in range(256)}
        mergeable_rank = {}

        if n_parallel is None or n_parallel == 1 or n_parallel == -1: # Implement single process logic.
            print(f'Training with single process')
            for i in tqdm(range(n_steps), desc="Traning Tokenier..."):
                stats = self._get_stats_chunk(ids_dict)
                pair = max(stats, key=stats.get)
                idx = i + 256

                mergeable_rank[pair] = idx
                vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

                ids_dict = self._merge_dict(ids_dict, pair, idx)
        else:
            print(f'Traning with {n_parallel} processes.')
            divisor, remainder = len(word_dict) // n_parallel, len(word_dict) % n_parallel
            task_queues = [multiprocessing.Queue() for _ in range(n_parallel)]
            result_queue = multiprocessing.Queue()

            ids_dict_kv = list(ids_dict.items())
            idx = 0
            processes = []
            for i in range(n_parallel):
                if i < remainder: # take one more chunk
                    partial_ids_dict = {k: v for k, v in ids_dict_kv[idx: idx + divisor + 1]}
                    idx = idx + divisor + 1
                else:
                    partial_ids_dict = {k: v for k, v in ids_dict_kv[idx: idx + divisor]}
                    idx = idx + divisor
                
                p = multiprocessing.Process(
                    target = self._parallel_train,
                    args = (i, partial_ids_dict, task_queues[i], result_queue)
                )
                p.start()
                processes.append(p)
            
            for i in tqdm(range(n_steps), desc=f'Training Tokenizer with {n_parallel} processes...'):
                for q in task_queues:
                    q.put(('COUNT', None))
            
                global_counter = Counter()
                for j in range(n_parallel):
                    global_counter.update(result_queue.get())
                pair = max(global_counter, key=global_counter.get)
                idx = i + 256

                for q in task_queues:
                    q.put(('MERGE', (pair, idx)))
                
                mergeable_rank[pair] = idx
                vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            for q in task_queues:
                q.put(('STOP', None))
            
        self.vocab = vocab
        self.mergeable_rank = mergeable_rank

                
    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}


    def _encode_ordinary(self, text):
        word_list = re.findall(self.compiled_pattern, text)
        ids_list = []
        for word in word_list:
            ids = list(word.encode('utf-8'))
            while len(ids) >= 2:
                merge_pair = min(ids, key = lambda x: self.mergeable_rank.get(x, float('inf')))
                if merge_pair not in self.mergeable_rank:
                    break
                ids = self.merge(ids, merge_pair, self.mergeable_rank[merge_pair])
            ids_list.append(ids)
        return ids_list


    def encode(self, text, allowed_special='none_raise'):
        # TODO Allowed special seems to have a gap with self.special_tokens used in decode (Maybe only to prevent user misuse special tokens??)
        if allowed_special == 'all':
            special = self.special_tokens
        elif allowed_special == 'none':
            special = {}
        elif allowed_special == 'none_raise':
            special = {}
            assert all(token not in text for token in self.special_tokens)
        else:
            raise ValueError(f'allowed_pecial={allowed_special} not understood')
        
        if not special:
            return self._encode_ordinary(text)
        
        special_pattern = '(' + '|'.join(re.escape(k) for k in special) + ')'
        special_chunks = re.split(special_pattern, text)

        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self._encode_ordinary(part))
        return ids

    def decode(self, ids):
        text_bytes = b''
        for id in ids:
            if id in self.vocab:
                text_bytes += self.vocab[id]
            elif id in self.special_tokens:
                text_bytes += self.inverse_special_tokens[id].encode('utf-8')
            else:
                raise ValueError(f'invalid token id: {id}')
        return text_bytes.decode('utf-8', errors='replace')

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("chatbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()