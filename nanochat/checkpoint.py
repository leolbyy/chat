import os
import json

import torch

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