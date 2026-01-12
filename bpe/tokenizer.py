from itertools import chain
# from multiprocessing import Queue, Process
import multiprocessing
from collections import Counter
import regex as re
import pyarrow.parquet as pq
import time


PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class BaseTokenizer():
    def __init__(self, pattern=None):
        self.pattern = pattern if pattern is not None else PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.vocab = {}
        self.mergeable_rank = {}
        self.special_tokens = {}


    # very likely this function should be moved to a common util file.
    # Implement here first for convinience.
    def _load_text(self, filepaths: dict):
        text = []
        for filepath, row_groups in filepaths.items():
            text.extend(pq.ParquetFile(filepath).read_row_groups(row_groups).column('text').to_pylist())
        text = ''.join(text)
        return text

    def _count_and_merge(self, worker_id, filepaths, task_queue, result_queue):
        print(f'Starting worker {worker_id}...')
        text = self._load_text(filepaths)
        word_list = re.findall(self.compiled_pattern, text)
        word_dict = Counter(word_list)
        print(f'Worker {worker_id}: Original text length is {len(text)}. Now processing {len(word_dict)} words')
        
        ids_dict = {tuple(k.encode('utf-8')): v for k, v in word_dict.items()}
        # ids_list = [list(word.encode('utf-8')) for word in word_list]

        while True:
            command, payload = task_queue.get()
            if command == 'STOP':
                pass
            elif command == 'CONTINUE':
                # merge if payload is not None
                if payload is not None:
                    pair, idx = payload
                    new_ids_dict = {}
                    for ids in ids_dict.keys():
                        newids = []
                        i = 0
                        while i < len(ids):
                            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                                newids.append(idx)
                                i += 2
                            else:
                                newids.append(ids[i])
                                i += 1
                        new_ids_dict[tuple(newids)] = ids_dict[ids]
                    ids_dict = new_ids_dict

                # get_stats
                # ids_zip_list = (zip(ids, ids[1:]) for ids in ids_dict.keys())
                # ids_flat_steam = chain.from_iterable(ids_zip_list)
                # stats = Counter(ids_flat_steam)
                stats = Counter()
                for ids in ids_dict.keys():
                    key_counter = Counter(zip(ids, ids[1:]))
                    key_counter = {k: v * ids_dict[ids] for k,v in key_counter.items()}
                    stats.update(key_counter)

                result_queue.put(stats)
            else:
                raise ValueError(f'command {command} not understood')


    def train_from_text(self, data_splits, vocab_size, n_parallel=8):
        task_queues = [multiprocessing.Queue() for _ in range(n_parallel)]
        result_queue = multiprocessing.Queue()

        processes = []
        for i in range(n_parallel):
            p = multiprocessing.Process(
                target = self._count_and_merge,
                args = (i, data_splits[i], task_queues[i], result_queue)
            )
            p.start()
            processes.append(p)
        
        assert vocab_size >= 256
        n_steps = vocab_size - 256

        vocab = {idx: bytes([idx]) for idx in range(256)}
        mergeable_rank = {}
        pair, idx = None, 0
        for i in range(n_steps):
            start_time = time.time()
            global_counter = Counter()
            print(f'start {i}/{n_steps} step...')

            if pair is None:
                for q in task_queues:
                    q.put(('CONTINUE', None))
            else:
                for q in task_queues:
                    q.put(('CONTINUE', (pair, idx)))

            for _ in range(n_parallel):
                partial_counter = result_queue.get()
                global_counter.update(partial_counter)
            
            pair = max(global_counter, key=global_counter.get)
            idx = idx + 256

            # update mergeable_rank
            mergeable_rank[pair] = idx
            # update vocab
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            end_time = time.time()
            print(f'One step times {end_time - start_time} seconds when using {n_parallel} processes.')
        for q in task_queues:
            q.put(('STOP', None))
        self.vocab = vocab
        self.mergeable_rank = mergeable_rank
        



    # def generate_chunk(self, ids_list, chunk_size=500):
    #     for i in range(0, len(ids_list), chunk_size):
    #         yield ids_list[i: i + chunk_size]

    # def get_stats(self, ids):
    #     count = {} # {(int, int): int}
    #     for id_pair in zip(ids, ids[1:]):
    #         count[id_pair] = count.get(id_pair, 0) + 1
    #     return count
    
    # def get_chunk_stats(self, ids_list):
    #     """
    #     Optimize for chunk count.
    #     """
    #     ids_flat = (zip(ids, ids[1:]) for ids in ids_list)
    #     ids_flat_stream = chain.from_iterable(ids_flat)

    #     return Counter(ids_flat_stream)


    # def get_stats_parallel(self, ids_list, n_parallel=8):
    #     if n_parallel is None or n_parallel == -1 or n_parallel == 1: # parallel processing is disabled
    #         return self.get_chunk_stats(ids_list)

    #     stats_final = Counter()
    #     with Pool(processes=n_parallel) as pool:
    #         stats_iterator = pool.imap_unordered(
    #             self.get_chunk_stats,
    #             self.generate_chunk(ids_list, chunk_size=3000),
    #             chunksize=1
    #         )
    #         for stats in stats_iterator:
    #             stats_final.update(stats)

    #     return stats_final


    # def merge(self, ids, pair, idx):
    #     newids = []
    #     i = 0
    #     while i < len(ids) - 1:
    #         if (ids[i], ids[i + 1]) == pair:
    #             newids.append(idx)
    #             i += 2
    #         else:
    #             newids.append(ids[i])
    #             i += 1
    #     return newids
    
    # def merge_parallel(self, ids_list, pair, idx, n_parallel=8):
    #     if n_parallel is None or n_parallel == 1 or n_parallel == -1:
    #         newids_list = []
    #         for ids in ids_list:
    #             newids_list.append(self.merge(ids, pair, idx))
    #         return newids_list
        
    #     with Pool(processes=n_parallel) as pool:
    #         newids_list = list(pool.starmap(self.merge, [(ids, pair, idx) for ids in ids_list], chunksize=500))
    #     return newids_list


    # def train_from_text(self, text, vocab_size, n_parallel = 8):
    #     """
    #     Text should be a list of strings
    #     """
    #     assert vocab_size >= 256 # should at least contain all the bytes
    #     n_steps = vocab_size - 256

    #     vocab = {idx: bytes([idx]) for idx in range(256)}
    #     mergeable_rank = {}

    #     text = [list(i.encode('utf-8')) for i in text]

    #     for i in range(n_steps):
    #         print(f'Now in {i}/{n_steps} steps!')
    #         stats = self.get_stats_parallel(text, n_parallel)
    #         pair = max(stats, key=stats.get)
    #         idx = i + 256
    #         # update mergeable_rank
    #         mergeable_rank[pair] = idx
    #         # update vocab
    #         vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

    #         text = self.merge_parallel(text, pair, idx, n_parallel)
        
        
        
    #     self.mergeable_rank = mergeable_rank
    #     self.vocab = vocab

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

    def load_from_directory(self, vocab_path, mergeable_rank_path):
        pass

    def save(self, base_dir):
        pass
