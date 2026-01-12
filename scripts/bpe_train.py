import os
import time
import argparse
from utils.common import get_base_dir
from utils.dataloader import load_text
from bpe.tokenizer import BaseTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start tokenizer training')
    parser.add_argument('-n', '--num-workers', type=int, default=-1, help='Number of workers to do parallel training. -1 = disable. (Default: -1)')
    parser.add_argument('--max-chars', type=int, default=1e9, help='Maximum chars to use for Tokenizer training. (Default: 1e9)')
    parser.add_argument('--vocab-size', type=int, default=32678, help='Vocabulary size (Default: 32678=2^15)')

    args = parser.parse_args()

    tokenizer = BaseTokenizer()
    start_time = time.time()
    tokenizer.train_from_text(load_text(), vocab_size=args.vocab_size, max_char=args.max_chars, n_parallel=args.num_workers)
    end_time = time.time()
    print(f'tokenizer finished training! Total time used is {end_time - start_time} secs!')
    tokenizer.save(base_dir=os.path.join(get_base_dir(), 'tokenizer'))
