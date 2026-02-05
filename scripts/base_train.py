"""
First Version without distributed training.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import time
from contextlib import nullcontext

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.common import get_base_dir, compute_init, autodetect_device_type
from utils.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit

from bpe.tokenizer import get_tokenizer, get_token_bytes

from nanochat.gpt import GPT, GPTConfig
from nanochat.loss_eval import evaluate_bpb
from nanochat.core_eval import evaluate_model
from nanochat.sample_eval import get_response
from nanochat.checkpoint import save_checkpoint, load_checkpoint

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model architecture
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention, need be devisible by model_dim")
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
# Training horizon (only one used, in order of precedence)
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
parser.add_argument("--target-param-data-ratio", type=int, default=8, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--tokens-per-step", type=int, default=524288, help="num of tokens to accumulate gradient before update param")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.2, help="cautious weight decay for the Muon optimizer (for weights)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding")
parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for embedding/unembedding")
parser.add_argument("--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup")
parser.add_argument("--warmdown-ratio", type=float, default=0.4, help="ratio of iterations for LR warmdown")
parser.add_argument("--final-lr-frac", type=float, default=0.0, help="final LR as fraction of initial LR")
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=20*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
args = parser.parse_args()
user_config = vars(args).copy() 

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
print(f"Device type set to {device_type}")
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type=device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# Tokenizer init
BASE_DIR = get_base_dir()
tokenizer_dir = os.path.join(BASE_DIR, 'tokenizer')
tokenizer = get_tokenizer(tokenizer_dir)
vocab_size = tokenizer.get_vocab_size()
print(f"Tokenizer vocab size: {vocab_size}")
token_bytes = get_token_bytes(tokenizer_dir, device=device)

# Model kwargs
model_dim = args.depth * args.aspect_ratio
assert model_dim % args.head_dim == 0, f"head_dim is not devisible by model_dim. head_dim: {args.head_dim}, model_dim: {model_dim}"
num_heads = model_dim // args.head_dim
num_kv_heads = num_heads # disable gqa. Allow config later
print(f"num_layers: {args.depth}")
print(f"model_dim: {model_dim}")
print(f"num_heads: {num_heads}")
print(f"num_kv_heads: {num_kv_heads}")

# Optimizer setup
tokens_per_fwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwd = tokens_per_fwd * ddp_world_size
assert args.tokens_per_step % world_tokens_per_fwd == 0
grad_accum_steps = args.tokens_per_step // world_tokens_per_fwd
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
model_config_kwargs = dict(seq_len=args.max_seq_len, vocab_size=vocab_size, n_layer=args.depth, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device) # All tensors got storage on target device but with garbage data (any data that was previously in the allocated memory)
model.init_buffer() # init RoPE

# If resuming, overwrite model parameters
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"
checkpoint_dir = os.path.join(BASE_DIR, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print(f'Resume from step {args.resume_from_step}')
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device=device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data
else:
    model.init_weights()


orig_model = model
model = torch.compile(model, dynamic=False)
num_params = sum(p.numel() for p in model.parameters())
num_scaling_params = orig_model.num_scaling_params()
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
    num_iterations = round(num_scaling_params * args.target_param_data_ratio / args.tokens_per_step)
    print(f"Calculated number of iterations using target param data ratio: {num_iterations}")
else:
    raise ValueError("Number of itertaions cannot be determined. Please specify one of (num-iterations, target-flops, target-param-data-ratio)")
total_tokens = args.tokens_per_step * num_iterations
print(f"Total number of training tokens: {total_tokens}")
print(f"Tokens : Param Ratio: {args.tokens_per_step * num_iterations / num_scaling_params:.2f}")
print(f"Total training FLOPs estimate: {total_tokens * num_flops_per_token:e}")


# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
adam_betas = (args.adam_beta1, args.adam_beta2)
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
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split='train', device=device, resume_state_dict=data_loader_resume_state_dict)
val_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split='val', device=device)
x, y, dataloader_state_dict = next(train_loader)

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# LR scheduler
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
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
    return weight_decay_scaled * (1 - it / num_iterations)

# Logging with tensorboard setup
if master_process:
    logging_tag = 'base_train'
    logging_dir = os.path.join(BASE_DIR, 'logs', logging_tag)
    writer = SummaryWriter(log_dir=logging_dir)


# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

# Handle this part later. Need to first understand what loop state is.
if not resuming:
    step = 0
    val_bpb = None
    min_val_bpb = float('inf')
    total_training_time = 0
    smooth_train_loss = 0
    ema_beta = 0.9
else:
    step = meta_data['step']
    val_bpb = meta_data['val_bpb']
    min_val_bpb = meta_data['loop_state']['min_val_bpb']
    total_training_time = meta_data['loop_state']['total_training_time']
    smooth_train_loss = meta_data['loop_state']['smooth_train_loss']
    ema_beta = meta_data['loop_state']['ema_beta']


# -----------------------------------------------------------------------------
# Training loop
while True:
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * args.tokens_per_step * step

    # eval val bpb
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        if master_process:
            writer.add_scalar(f'{logging_tag}/bpb', val_bpb, step)
        print(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        model.train()
    
    # estimate CORE metric with original model to avoid re-compilation (input shape constantly changes)
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0) and step % args.core_metric_every == 0):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        if master_process:
            writer.add_scalars(f'{logging_tag}/eval_results', results['results'], step)
            writer.add_scalars(f'{logging_tag}/eval_centered_results', results['centerd_results'], step)
            writer.add_scalar(f'{logging_tag}/core_metric', results['core_metric'], step)
        print(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        model.train()
    

    # sample from model. Only on master process
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        for prompt in prompts:
            with autocast_ctx:
                response = get_response(model, tokenizer, prompt, max_tokens=16)
            print(f'Input prompt: {prompt} --> Response: {response}')
            writer.add_text(f'{logging_tag}/samples', f'Input prompt: {prompt} --> Response: {response}', step)
        model.train()
    
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                'step': step,
                'val_bpb': val_bpb,
                'model_config': model_config_kwargs,
                'user_config': user_config,
                'device_batch_size': args.device_batch_size,
                'max_seq_len': args.max_seq_len,
                'dataloader_state_dict': dataloader_state_dict,
                'loop_state': {
                    'min_val_bpb': min_val_bpb,
                    'total_training_time': total_training_time,
                    'smooth_train_loss': smooth_train_loss,
                    'ema_beta': ema_beta
                }
            },
            rank=ddp_rank
        )
    
    if last_step:
        break

    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        # computational graph will be saved until loss.backward is called, so we cannot sum loss and do average outside for loop
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader)
    
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group['lr'] = group['initial_lr'] * lrm
    
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in muon_optimizer.param_groups:
        group['momentum'] = muon_momentum
        group['weight_decay'] = muon_weight_decay
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True) # equivalant of calling optim.zero_grad() for all optimizers. set_to_none will use less memory.

    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # -------------------------------------------------------------------------

    # logging (CPU action only)
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(args.tokens_per_step / dt)
    flops_per_sec = num_flops_per_token * args.tokens_per_step / dt
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = dataloader_state_dict["epoch"]
    if master_process:
        writer.add_scalar(f'{logging_tag}/debiased_train_loss', debiased_smooth_loss, step)
        writer.add_scalar(f'{logging_tag}/tok-per-sec', tok_per_sec, step)
    print(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.2f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")

    # state update
    step += 1

# print a few more stats
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
print(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print(f"Minimum validation bpb: {min_val_bpb:.6f}")
    

