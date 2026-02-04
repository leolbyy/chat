import os
import torch
import pyarrow.parquet as pq
from utils.common import get_base_dir, get_dist_info


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


# def load_text(split, pf=0, start=0, step=1):
#     parquet_paths = list_parquet_files(DATA_DIR)
#     for parquet_path in parquet_paths:
#         pf = pq.ParquetFile(parquet_path)
#         for i in range(start, pf.num_row_groups, step):
#             text = pf.read_row_group(i).column('text').to_pylist()
#             yield text


def load_text(split, pf_idx=0, start=0, step=1, epoch=0, tokenizer_batch_size=128):
    parquet_paths = list_parquet_files()
    if split == 'train':
        parquet_paths = parquet_paths[:-1]
    else:
        parquet_paths = parquet_paths[-1:]
    
    while True:
        if pf_idx >= len(parquet_paths):
            pf_idx = 0
        while pf_idx < len(parquet_paths):
            pf = pq.ParquetFile(parquet_paths[pf_idx])
            for rg_idx in range(start, pf.num_row_groups, step):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i: i + tokenizer_batch_size], (pf_idx, rg_idx, epoch)
            pf_idx += 1
        pf_idx = 0
        epoch += 1



def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer,
    B, T, split,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device="cuda",
    resume_state_dict=None,
    buffer_size=1000
):

    assert split in ('train', 'val'), f"split must be 'train' or 'val'. got {split}"

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if resume_state_dict is not None:
        pf_idx = resume_state_dict['pf_idx']
        rg_idx = resume_state_dict['rg_idx'] + ddp_rank
        epoch = resume_state_dict['epoch']
    else:
        pf_idx = 0
        rg_idx = ddp_rank
        epoch = 0
    
    step = ddp_world_size
    
    row_capacity = T + 1
    batches = load_text(split, pf_idx=pf_idx, start=rg_idx, step=step, epoch=epoch, tokenizer_batch_size=tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []

    while True:
        rows = []
        for _ in range(B):
            row = []
            while len(row) < row_capacity:
                while len(doc_buffer) < buffer_size:
                    doc_batch, (pf_idx, rg_idx, epoch) = next(batches)
                    token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
                    doc_buffer.extend(token_lists)
                remaining = row_capacity - len(row)
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    if len(doc) <= remaining and len(doc) > best_len:
                        best_idx = i
                        best_len = len(doc)
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row.extend(doc)
                else: # no doc can fit in the remaining space
                    shortest_idx = min(range(len(doc_buffer)), key=lambda x: len(doc_buffer[x]))
                    doc = doc_buffer.pop(shortest_idx)
                    row.extend(doc[:remaining])
            rows.append(row)
        
        use_cuda = device == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, non_blocking=use_cuda)
        yield inputs, targets, {"pf_idx": pf_idx, "rg_idx": rg_idx, "epoch": epoch}


def get_task_example(dataset, task_idx, idx, step, progress):
    num_examples = dataset.get_num_examples()
    num_exmaples_total = dataset.get_num_examples_total()
    epoch = 0
    while True:
        examples = []
        if idx >= num_examples[task_idx]:
            idx = idx - num_examples[task_idx]
            task_idx += 1
            if task_idx >= len(num_examples):
                task_idx = 0
                epoch += 1
        messages = dataset[task_idx].get_example(idx)
        idx += step
        progress += step / num_exmaples_total
        yield messages, epoch, progress



def distributed_task_data_loader(
    tokenizer,
    B, T,
    dataset,
    split,
    device="cuda",
    buffer_size=1000
):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    task_idx = 0
    idx = ddp_rank
    epoch = 0
    step = ddp_world_size
    
    row_capacity = T + 1
    example_iterator = get_task_example(dataset, task_idx, idx, step, 0.0)
    doc_buffer = []

    while True:
        rows = []
        for _ in range(B):
            row = []
            while len(row) < row_capacity:
                while len(doc_buffer) < buffer_size:
                    messages, epoch, progress = next(example_iterator)
                    token_ids, _ = tokenizer.render_conversation(messages)
                    doc_buffer.append(token_ids)
                remaining = row_capacity - len(row)
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    if len(doc) <= remaining and len(doc) > best_len:
                        best_idx = i
                        best_len = len(doc)
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row.extend(doc)
                else: # no doc can fit in the remaining space
                    shortest_idx = min(range(len(doc_buffer)), key=lambda x: len(doc_buffer[x]))
                    doc = doc_buffer.pop(shortest_idx)
                    row.extend(doc[:remaining])
            rows.append(row)
        
        use_cuda = device == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, non_blocking=use_cuda)
        yield inputs, targets, epoch, progress 


def distributed_task_data_loader_with_pad(
    tokenizer,
    B, T,
    dataset,
    device="cuda"
):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    task_idx = 0
    idx = ddp_rank
    epoch = 0
    step = ddp_world_size
    
    row_capacity = T + 1
    example_iterator = get_task_example(dataset, task_idx, idx, step, 0.0)
    bos_token_id = tokenizer.get_bos_token_id()

    messages = None

    while True:
        rows = []
        row_lengths = []
        for _ in range(B):
            row = []
            row_length = 0
            while len(row) < row_capacity:
                remaining = row_capacity - len(row)
                if messages is None:
                    messages, epoch, progress = next(example_iterator)
                    token_ids, _ = tokenizer.render_conversation(messages)
                if len(token_ids) <= remaining:
                    row.extend(token_ids)
                    row_length += len(token_ids)
                    messages = None
                else: # pad with <|bos|>, since we can't afford to waste data due to small dataset size
                    row.extend([bos_token_id] * remaining)
            rows.append(row)
            row_lengths.append(row_length)
        
        use_cuda = device == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, non_blocking=use_cuda)
        for i, row_length in enumerate(row_lengths):
            targets[i, row_length - 1:] = -1
        print(inputs[0], targets[0], sep='\n')
        yield inputs, targets, epoch, progress # buggy here, progess may exceed real progress due to caching. (i.e. if new message does not fit, it will be cached for next row)




def distributed_sft_data_loader(
        tokenizer,
        B, T,
        dataset,
        device="cuda"
):
    # we assume no single conversation exceeds T tokens

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    task_idx = 0
    idx = ddp_rank
    epoch = 0
    step = ddp_world_size

    row_capacity = T + 1
    example_iterator = get_task_example(dataset, task_idx, idx, step, 0.0)
    bos_token_id = tokenizer.get_bos_token_id()

    messages = None
    
    while True:
        inputs = torch.full((B, T), bos_token_id, dtype=torch.long, device=device)
        targets = torch.full((B, T), -1, dtype=torch.long, device=device)
        # we cannot trim here because of model is compiled with dynamic=False. trim will cause re-compilation which is slow.
        # max_len = 0
        # for sft, each row is exactly one conversation
        # this is to mimic real usage where each conversation is processed independently
        for _ in range(B):
            messages, epoch, progress = next(example_iterator)
            token_ids, masks = tokenizer.render_conversation(messages)
            max_len = max(max_len, len(token_ids))

            inputs[_, :len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
            row_targets = torch.tensor(token_ids[1:], dtype=torch.long)
            mask_tensor = ~torch.tensor(masks[1:], dtype=torch.bool)  # 0 to True and 1 to False
            row_targets = torch.where(mask_tensor, -1, row_targets)
            targets[_, :len(token_ids) - 1] = row_targets
        # # trim to max_len
        # inputs = inputs[:, :max_len]
        # targets = targets[:, :max_len]

        yield inputs, targets, epoch, progress




























