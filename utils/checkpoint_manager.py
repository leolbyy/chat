import torch



def save_checkpoint(
    checkpoint_dir,
    step,
    model_data,
    optimizer_data,
    meta_data,
    rank=0
):
    if rank == 0: # only save on master process
        os.makedirs(checkpoint_dir, exists_ok=True)

        model_path = os.path.join(checkpoint_dir, f'model_{step:06d}.pt')
        torch.save(model_data, model_path)
        print(f"Saved model to {model_path}")

        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open('meta_data', 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=4)
        print(f"Saved metadata to {meta_path}")
    
    os.makedirs(checkpoint_dir, exists_ok=True) # in case worker process run here before master process create dir
    optimizer_path = os.path.join(checkpoint_dir, f'optmizer_{step:06d}_{rank:02d}.pt')
    torch.save(optimizer_data, optimizer_path)
    print(f"Saved optimizer state to {optimizer_path}")

