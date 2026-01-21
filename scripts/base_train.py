"""
First Version without distributed training.
"""

import os

import argparse
import time
from contextlib import nullcontext

import torch

from bpe.tokenizer import get_tokenizer()

from nanochat.gpt import GPT, GPTConfig


# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Runtime
parser.add_argument("--device_type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect_ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head_dim", type=int, default=128, help="target head dimension for attention, need be devisible by model_dim")
parser.add_argument("--max_seq_len", type=int, default=2048, help="max context length")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num_iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target_flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target_param_data_ratio", type=int, default=8, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device_batch_size", type=int, default=32, help="per-device batch size")
parser.add_argument("--tokens_per_step", type=int, default=524288, help="num of tokens to accumulate gradient before update param")
parser.add_argument("--embedding_lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding_lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight_decay", type=float, default=0.2, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix_lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar_lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--adam_beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam_beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--warmup_ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown_ratio", type=float, default=0.4, help="ratio of iterations for LR warmdown")
parser.add_argument("--final_lr_frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume_from_step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval_every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval_tokens", type=int, default=20*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core_metric_every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core_metric_max_per_task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample_every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save_every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model_tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()


# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
print(f"Device type set to {device_type}")
ddp, ddp_rank, ddp_local_rank, ddp_world_size = compute_init(device_type=device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device)
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# Tokenizer init
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, 'tokenizer')
tokenizer = get_tokenizer(tokenizer_dir)
vocab_size = tokenizer.get_vocab_size()
print(f"Tokenizer vocab size: {vocab_size}")

# Model kwargs
model_dim = args.depth * args.aspect_ratio
assert model_dim % head_dim == 0, f"head_dim is not devisible by model_dim. head_dim: {args.head_dim}, model_dim: {model_dim}"
num_heads = model_dim // head_dim
num_kv_heads = num_heads # disable gqa. Allow config later
print(f"num_layers: {args.depth}")
print(f"model_dim: {model_dim}")
print(f"num_heads: {num_heads}")
print(f"num_kv_heads: {num_kv_heads}")

# Optimizer setup
tokens_per_fwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwd = tokens_per_fwd * ddp_world_size
assert args.tokens_per_step % world_tokens_per_fwd == 0
grad_accum_steps = args.tokens_per_fwd // world_tokens_per_fwd
print(f"Tokens / micro-batch / rank: {args.device_batch_size} * {args.max_seq_len} = {tokens_per_fwd}")
print(f"Tokens / micro-batch: {world_tokens_per_fwd}")
print(f"Total batch size in tokens {args.tokens_per_step} --> gradient accumulation steps: {grad_accum_steps}")

# Batch size scaling for learning rate
batch_lr_scale = 1.0
reference_batch_size = 2 ** 19
batch_ratio = args.tokens_per_step / reference_batch_size
if batch_ratio != 1.0:
    # SGD use liner scaling as it only use first order derivate
    # AdamW use sqrt as when batch size is k times larger, the standard deivation is 1/sqrt(k) * original_sd.
    # There for times sqrt batch_lr_scale to match with original deviation
    batch_lr_scale = batch_ratio ** 0.5
    print(f"Scaling learning rates by {batch_lr_scale:.4f} for tokens {args.tokens_per_step}. (Reference: {reference_batch_size})")

weight_decay_scaled = args.weight_decay * (12 / args.depth) ** 2
if args.depth != 12:
    print(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")



# -----------------------------------------------------------------------------
# Initialize the Model

# Create a new model with random weights
model_kwargs_config = dict(sequence_len=args.max_seq_len, vocab_size=vocab_size, n_layer=args.depth, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)
with torch.device("meta"):
    model_cofig = GPTConfig(**model_kwargs_config)
    model = GPT(model_config)
model.to_empty(device=device) # All tensors got storage on target device but with garbage data (any data that was previously in the allocated memory)
model.init_weights()

# If resuming, overwrite model parameters
# Implement later

origi_model = model
model = torch.compile(model, dynamic=False)
num_params = sum(p.numel() for p in model.parameters())
print(model.parameters())
num_scaling_params = origi_model.num_scaling_params()
print(f"Number of parameters after compile: {num_params} (original: {num_scaling_params})")
num_flops_per_token = model.estimate_flops()
print(f"Estimated flops per token: {num_flops_per_token}")


if args.num_iterations > 0:
    num_iterations = args.num_iterations
    print(f"Using user provided number of iterations: {args.num_iterations}")
elif args.target_flops > 0:
    num_iterations = round(args.target_flops / (num_flops_per_token * args.tokens_per_step))
    print(f'Using taget flops to get number of iterations: {num_iterations}')
elif args.target_param_data_ratio > 0:
    num_iterations = round(num_scaling_params * target_param_data_ratio / args.tokens_per_step)
    print(f"Calculated number of iterations using target param data ratio: {num_iterations}")
else:
    raise ValueError("Number of itertaions cannot be determined. Please specify one of (num-iterations, target-flops, target-param-data-ratio)")
total_tokens = args.tokens_per_step * num_iterations
print(f"Total number of training tokens: {total_tokens}")
print(f"Tokens : Param Ratio: {args.tokens_per_step * num_iterations / num_scaling_params:.2f}")
print(f"Total training FLOPs estimate: {total_tokens * num_flops_per_token:e}")


# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
adma_betas = (args.adam_beta1, args.adam_beta2)
optimizers = model.setup_optimizers(
    unembedding_lr = args.unembedding_lr * batch_lr_scale,
    embedding_lr = args.embedding_lr * batch_lr_scale,
    matrix_lr = args.matrix_lr * batch_lr_scale,
    weight_decay = args.weight_decay * weight_decay_scaled,
    adam_betas = adam_betas,
    scalar_lr = args.scalar_lr * batch_lr_scale
)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
data_loader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split='train', device=device, resumt_state_dict=data_loader_resume_state_dict)
val_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split='val', deivce=device)
x, y, dataloader_state_dict = next(trian_loader)

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# LR scheduler
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_iters * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac
    
# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# Weight decay scheduler for muon
def get_weight_decay(it):
    return ewight_decay_scaled * (1 - it / num_iterations)



# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

# Handle this part later. Need to first understand what loop state is.
# TODO

# -----------------------------------------------------------------------------
# Training loop
while True:
    
