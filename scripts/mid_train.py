import os

import argparse
import time
from contextlib import nullcontext

import torch

from utils.common import get_base_dir, compute_init
from utils.dataloader import distributed_task_data_loader
from utils.checkpoint import save_checkpoint, load_checkpoint, load_model_from_dir

from bpe.tokenizer import get_tokenizer, get_token_bytes

from nanochat.gpt import GPT, GPTConfig
from nanochat.loss_eval import evaluate_bpb
from nanochat.core_eval import evaluate_model
from nanochat.sample_eval import get_response
from nanochat.tasks import TaskMixture, GSM8K, SmolTalk, MMLU, CustomJSON, SimpleSpelling, SpellingBee

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Midtrain the model")
# Runtime
parser.add_argument("--device-type", type=str, default="", choices={'cuda', 'cpu', 'mps'}, help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch)")
# Batch sizes
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--tokens-per-step", type=int, default=524288, help="tokens to process for each step")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=1.0, help="initial LR as fraction of base LR")
# Evaluation
parser.add_argument("--eval-every", type=int, default=150, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=20*524288, help="number of tokens to evaluate val loss on")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
print(f'Deivce type set to {device_type}')

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type=device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
synchronize = torch.cuda.synchronize if device_type == 'cuda' else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == 'cuda' else lambda: 0

# tokenizer init
BASE_DIR = get_base_dir()
tokenizer_dir = os.path.join(BASE_DIR, 'tokenizer')
tokenizer = get_tokenizer(tokenizer_dir)
token_bytes = get_token_bytes()

# load model
model, model_config_kwargs = load_model_from_dir(checkpoint_dir)

orig_model = model
model = torch.compile(model, dynamic=False)
depth = model.config.n_layer
num_params = sum(p.numel() for p in model.parameters())
print(model.parameters())
num_scaling_params = orig_model.num_scaling_params()
print(f"Number of parameters after compile: {num_params} (original: {num_scaling_params})")
num_flops_per_token = model.estimate_flops()
print(f"Estimated flops per token: {num_flops_per_token}")

# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
adam_betas = (args.adam_beta1, args.adam_beta2)
optimizers = model.setup_optimizers(
    unembedding_lr = args.unembedding_lr,
    embedding_lr = args.embedding_lr,
    matrix_lr = args.matrix_lr,
    weight_decay = args.weight_decay,
)
adamw_optimizer, muon_optimizer = optimizers
for opt in optimizers:
    for group in opt.param_groups:
        group['lr'] = group['lr'] * args.init_lr_frac
        group['init_lr'] = group['lr']

tokens_per_fwd = args.device_batch_size * args.max_seq_len
world_tokens_per_fwd = tokens_per_fwd * ddp_world_size
assert args.tokens_per_step % world_tokens_per_fwd == 0
grad_accum_steps = args.tokens_per_step // world_tokens_per_fwd
print(f"Tokens / micro-batch / rank: {args.device_batch_size} * {args.max_seq_len} = {tokens_per_fwd}")
print(f"Tokens / micro-batch: {world_tokens_per_fwd}")
print(f"Total batch size in tokens {args.tokens_per_step} --> gradient accumulation steps: {grad_accum_steps}")

# Get tokenizer
tokenizer_dir = os.path.join(BASE_DIR, 'tokenizer')
tokenizer = get_tokenizer(tokenizer_dir)
token_bytes = get_token_bytes(tokenizer_dir, device=device)


# define dataset
personality_filepath = os.path.join(BASE_DIR, 'identity_conversations.jsonl')
train_dataset = TaskMixture([
    SmolTalk(split='train'),
    MMLU(subset='auxiliary_train', split='train'),
    GSM8K(subset='main', split='train'),
    CustomJSON(filepath=personality_filepath),
    CustomJSON(filepath=personality_filepath),
    SimpleSpelling(split='train'),
    SpellingBee(size=10000, split='train')
])
val_dataset = TaskMixture([
    SmolTalk(split='test'),
    MMLU(subset='all', split='test'),
    GSM8K(subset='main', split='test')
])


train_loader = distributed_task_data_loader(tokenizer, args.device_batch_size, args.max_seq_len, dataset=train_dataset, device=device)
val_loader = distributed_task_data_loader(tokenizer, args.device_batch_size, args.max_seq_len, dataset=val_dataset, device=device)
x, y, epoch = next(train_loader)

min_val_bpb = float('inf')
smooth_train_loss = 0
ema_beta = 0.9
total_training_time =  0
step = 1

progress = 0.0

def get_lr_multiplier(progress):
    return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

while True:
    flops_so_far = num_flops_per_token * args.tokens_per_step * step

    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        model.train()
    
    if master_process and last_step:
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(BASE_DIR, 'mid_train', output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            None,
            {
                'step': step,
                'val_bpb': val_bpb,
                'model_config': model_config_kwargs,
                'user_config': user_config
            }
        )
    if last_step:
        break
    
    synchronize()
    start_time = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch, data_progress = next(train_loader)
    
    # if num-iterations is not given, we use progress from dataloader
    # else, we use progress calcuated by num-iteration
    if args.num_iterations is None:
        progress = data_progress
    else:
        progress = step / args.num_iterations

    lrm = get_lr_multiplier(progress)
    for opt in optmizers:
        for group in opt.param_groups:
            group['lr'] - group['inital_lr'] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group['momentum'] = muon_momentum
    for opt in optimizers:
        opt.step()
    
    model.zero_grad(set_to_none=True)

    synchronize()
    end_time = time.time()
    dt = end_time - start_time


    
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step))
    tok_per_sec = args.tokens_per_step / dt
    flops_per_sec = num_flops_per_token * args.tokens_per_step / dt
    if step > 10:
        total_training_time += dt
    
    print(f"Step {step:05d} {progress * 100:2f}% | loss: {debiased_smooth_loss} | lrm: {lrm:.2f} | dt: {dr * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | epoch: {epoch} | total time: {total_trianing_time/60:.2f}m")


    step += 1
    if step == args.num_iterations or (master_process and progress >= 1.0):
        last_step = True