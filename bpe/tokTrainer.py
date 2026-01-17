import regex as re
import multiprocessing
from tqdm import tqdm
from collections import Counter

from utils.dataloader import load_text

class TokenizerTrainer:
    def __init__(self, vocab_size, max_char, pattern, special_tokens, n_parallel=None):
        self.vocab_size = vocab_size
        self.max_char = max_char
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern)
        self.special_tokens = special_tokens
        self.n_parallel = n_parallel
    
    def _load_text(self):
        max_char = self.max_char

        char_count = 0
        word_dict = Counter()
        for text in load_text():
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

    def _register_special_tokens(self, special_tokens, start_index=0):
        assert isinstance(special_tokens, list)
        return {k: v + start_index for v, k in enumerate(special_tokens)}

    def train_from_iterator(self):
        vocab_size = self.vocab_size
        max_char = self.max_char
        n_parallel = self.n_parallel
        special_tokens = self.special_tokens
        pattern = self.pattern
        
        assert vocab_size >= 256
        n_steps = vocab_size - 256

        word_dict = self._load_text()
        ids_dict = {tuple(word.encode('utf-8')): v for word, v in word_dict.items()}

        vocab = {idx: bytes([idx]) for idx in range(256)}
        vocab_inverse_ids = {}

        if n_parallel is None or n_parallel == 1 or n_parallel == -1: # Implement single process logic.
            print(f'Training with single process')
            for i in tqdm(range(n_steps), desc="Traning Tokenier..."):
                stats = self._get_stats_chunk(ids_dict)
                pair = max(stats, key=stats.get)
                idx = i + 256

                vocab_inverse_ids[pair] = idx
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
                
                vocab_inverse_ids[pair] = idx
                vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            for q in task_queues:
                q.put(('STOP', None))
            
        special_tokens = self._register_special_tokens(special_tokens, start_index=max(vocab) + 1)
        mergeable_ranks = {v: k for k, v in vocab.items()}

        return mergeable_ranks, special_tokens, pattern