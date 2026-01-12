import pyarrow.parquet as pq
from utils.common import get_base_dir
import os
from bpe.tokenizer import BaseTokenizer

BASE_DIR = get_base_dir()
DATA_DIR = os.path.join(BASE_DIR, 'data')

def list_parquet_files(data_dir=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def split_data(n_parallel = 8):
    parquet_files = list_parquet_files()
    row_groups = []
    n_groups = 0
    for parquet_file in parquet_files:
        pf = pq.ParquetFile(parquet_file)
        n_groups += pf.num_row_groups
        row_groups.extend([(parquet_file, i) for i in range(pf.num_row_groups)])

    # split the data to n_parallel parts
    divisor, remainder = n_groups // n_parallel, n_groups % n_parallel
    splits = []
    idx = 0
    for i in range(n_parallel):
        part = {}
        if i < remainder: # handle one more data row group
            for filepath, row_group in row_groups[idx: idx + divisor + 1]:
                if filepath not in part:
                    part[filepath] = [row_group]
                else:
                    part[filepath].append(row_group)
            splits.append(part)
            idx = idx + divisor + 1
        else:
            for filepath, row_group in row_groups[idx: idx + divisor]:
                if filepath not in part:
                    part[filepath] = [row_group]
                else:
                    part[filepath].append(row_group)
            splits.append(part)
            idx = idx + divisor
    return splits
    




# def load_text():
#     parquet_paths = list_parquet_files(DATA_DIR)[:2]
#     text = []
#     for parquet_path in parquet_paths:
#         partial_list = pq.ParquetFile(parquet_path).read().column('text').to_pylist()
#         text.extend(partial_list)
#     return text




# text = load_text()
n_parallel = 5

data_splits = split_data(n_parallel=n_parallel)

tokenizer = BaseTokenizer()
tokenizer.train_from_text(data_splits, vocab_size=2048, n_parallel=n_parallel)

a = tokenizer.encode('This is a test')
print(a)


b = tokenizer.decode(a)
print(b)
print(b == 'This is a test')