import math
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate_bpb(model, val_iter, steps, token_bytes):
    # bits --> entropy with log2
    # nats --> entropy with ln (default in most loss functions)
    # bans --> entropy with log10
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    for _ in range(steps):
        x, y, _ = next(val_iter)
        loss2d = model(x, y, loss_reduction='none')
        loss = loss2d.view(-1)
        y = y.view(-1)
        assert (y >= 0).all(), f"How come there is a target has token id less than 1???? {y}"
        if (y < 0).any():
            print(y)
            print(torch.min(y))

        num_bytes = token_bytes[y]
        total_nats += (loss * (num_bytes > 0)).sum() # in tokenizer, bytes count for special tokens are 0. By > 0, we ignore all the special tokens is any
        total_bytes += num_bytes.sum()

    world_size = dist.get_world_size if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()

    if total_bytes == 0:
        return float('inf')
    bpb = (total_nats / math.log(2)) / total_bytes
    return bpb

