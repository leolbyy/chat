import os
import re
import json

import torch
from nanochat.gpt import GPTConfig, GPT

def save_checkpoint(
    checkpoint_dir,
    step,
    model_state_dict,
    optimizer_state_dict_list,
    metadata,
    rank
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        
        model_path = os.path.join(checkpoint_dir, f'model_{step:06d}.pt')
        torch.save(model_state_dict, model_path)

        meta_path = os.path.join(checkpoint_dir, f'meta_{step:06d}.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        
    # optimizer state is NOT shared across rank due to ZeRo-2 style distributed optmizer
    optimizer_path = os.path.join(checkpoint_dir, f'optimizer_{step:06d}_rank{rank:02d}.pt')
    torch.save(optimizer_state_dict_list, optimizer_path)


def load_checkpoint(
    checkpoint_dir,
    step,
    device,
    load_optimizer,
    rank
):
    model_path = os.path.join(checkpoint_dir, f'model_{step:06d}.pt')
    model_data = torch.load(model_path, map_location=device)
    if load_optimizer: # for training
        optimizer_path = os.path.join(checkpoint_dir, f'optimizer_{step:06d}_rank{rank:02d}.pt')
        optimizer_data = torch.load(optimizer_path, map_location=device)
    else:
        optimizer_data = None
    meta_path = os.path.join(checkpoint_dir, f'meta_{step:06d}.json')
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
    
    return model_data, optimizer_data, meta_data

def _find_latest_checkpoint(checkpoint_dir):
    files = os.listdir(checkpoint_dir)
    pattern = re.compile(r"model_(\d+)\.pt")
    max_step = None
    for file in files:
        match = pattern.search(file)
        if match:
            step = int(match.group(1))
            if max_step is None:
                max_step = step
            elif step > max_step:
                max_step = step
            else:
                pass
    assert max_step is not None, f"Model checkpoints not found."
    return max_step




def load_model_from_dir(checkpoint_dir, device, rank):
    max_step = _find_latest_checkpoint(checkpoint_dir)
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, max_step, device, load_optimizer=False, rank=rank)
    model_config_kwargs = meta_data['model_config']
    with torch.device("meta"):
        model_config = GPTConfig(**model_config_kwargs)
        model = GPT(model_config)
    model.to_empty(device=device) # All tensors got storage on target device but with garbage data (any data that was previously in the allocated memory)
    model.init_buffer() # init RoPE

    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data

    return model, model_config_kwargs
