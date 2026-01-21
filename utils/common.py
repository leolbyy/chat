import os
import torch
import torch.distributed as dist

def get_base_dir():
    if 'CHAT_BASE_DIR' in os.environ:
        return os.environ.get('CHAT_BASE_DIR')
    home_dir = os.path.expanduser('~')
    base_dir = os.path.join(home_dir, 'learn', 'chat')
    return base_dir

def autodetect_device_type():
    if torch.cuda.is_available:
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_dist_info():
    if all(var in os.environ for var in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def compute_init(device_type="cuda"):
    assert device_type in ("cuda", "mps", "cpu"), f"Invalid device type: {device_type}"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Device type set to cuda but cuda is not available"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Device type set to mps but mps is not available"
    
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    
    if device_type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "tf32"

    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)
    
    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size
