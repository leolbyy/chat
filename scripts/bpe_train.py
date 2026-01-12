import pyarrow.parquet as pq
from utils.common import get_base_dir
import os
from bpe.tokenizer import BaseTokenizer
import sys


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



def load_text(start=0, step=1):
    parquet_paths = list_parquet_files(DATA_DIR)[:8]
    for parquet_path in parquet_paths:
        pf = pq.ParquetFile(parquet_path)
        for i in range(start, pf.num_row_groups, step):
            text = ''.join(pf.read_row_group(i).column('text').to_pylist())
            yield text




# text = load_text()
n_parallel = int(sys.argv[1])


tokenizer = BaseTokenizer()
tokenizer.train_from_text(load_text(), vocab_size=2048, max_char=1e8, n_parallel=n_parallel)

a = tokenizer.encode('This is a test')
print(a)


b = tokenizer.decode(a)
print(b)
print(b == 'This is a test')