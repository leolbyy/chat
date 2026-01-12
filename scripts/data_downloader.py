import os
import requests
import time
import argparse
import logging
from multiprocessing.pool import ThreadPool as Pool
import pyarrow.parquet as pq
from utils.common import get_base_dir



def download_single_file(index):
    filename = f'shard_{index:05d}.parquet'
    url = f'{BASE_URL}/{filename}'
    filepath = f'{DATA_DIR}/{filename}'
    tmp_filepath = f'{DATA_DIR}/{filename}.tmp'

    # check whether file already downloaded
    if os.path.exists(filepath):
        logger.info(f'File {filename} already exist. Skip download.')
    else:
        logger.debug(f'Downloading {filename}...')

    for attempt in range(MAX_ATTEMPTS):
        logger.debug(f'Start {attempt} time to download {filename}...')
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(tmp_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
            os.rename(tmp_filepath, filepath)
            # logger.debug(f'Successfully downloaded {filename}.')
            return True
        
        except (requests.RequestException, IOError) as e:
            # Remove temp file if it's there
            if os.path.exists(tmp_filepath):
                try:
                    os.remove(tmp_filepath)
                except:
                    pass
            if attempt < max_attempts:
                logger.debug(f'Download {filename} failed for {attempt}/{MAX_ATTEMPTS}: {e}.\nRetry in {WAIT_TIME} seconds.')
                time.sleep(WAIT_TIME)
                continue
            else:
                logger.warning(f'Download {filename} failed for all {MAX_ATTEMPTS}. Skipping.')
                return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download FineWeb-Edu 100BT dataset shards')
    parser.add_argument('-n', '--num-files', type=int, default=-1, help='Number of shards to download. -1 = disable. (Default: -1)')
    parser.add_argument('-t', '--num-threads', type=int, default=16, help='Number of threads to download the files. (default: 16)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    BASE_DIR = get_base_dir()
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    os.makedirs(DATA_DIR, exist_ok=True)
    BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
    MAX_SHARD = 1822
    MAX_ATTEMPTS = 5
    WAIT_TIME = 3

    num_files = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num_files))

    logger.info(f"Downloading {len(ids_to_download)} shards using {args.num_threads} threads...")
    with Pool(processes=args.num_threads) as pool:
        results = pool.map(download_single_file, ids_to_download)

    successful = sum(1 for success in results if success)
    logger.info(f'Done! Downloaded {successful}/{num_files} shards to {DATA_DIR}')   