import os

import argparse
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from utils.common import get_base_dir, compute_init, autodetect_device_type
from utils.dataloader import distributed_sft_data_loader
from utils.checkpoint import save_checkpoint, load_checkpoint, load_model_from_dir

from bpe.tokenizer import get_tokenizer, get_token_bytes

from nanochat.gpt import GPT, GPTConfig
from nanochat.loss_eval import evaluate_bpb
from nanochat.core_eval import evaluate_model
from nanochat.sample_eval import get_response
from nanochat.tasks import TaskMixture, GSM8K, SmolTalk, MMLU, CustomJSON, SimpleSpelling, SpellingBee, ARC, eval_task

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Midtrain the model")
# Runtime
parser.add_argument("--device-type", type=str, default="", choices={'cuda', 'cpu', 'mps'}, help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("--model-source",type=str, default='mid', choices={'base', 'mid'}, help="base|mid")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch)")
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs to train")
# Batch sizes
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--target-examples-per-step", type=int, default=32, help="examples to process for each step")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=1.0, help="initial LR as fraction of base LR")
# Evaluation
parser.add_argument("--eval-every", type=int, default=150, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=20*524288, help="number of tokens to evaluate val loss on")
parser.add_argument("--task-eval-every", type=int, default=150, help="evaluate tasks every N steps (-1 = disable)")
parser.add_argument("--max-problems-per-task", type=int, default=32, help="maximum number of problems to eval per task, -1 = disbaled")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------
if args.num_iterations == -1:
    assert args.num_epochs > 0, "Either num-iterations or num-epochs must be positive"

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
token_bytes = get_token_bytes(tokenizer_dir, device=device)

# load model
checkpoint_dir = os.path.join(BASE_DIR, f'{args.model_source}_checkpoints', args.model_tag) # TODO Finish model tag logic handling
model, model_config_kwargs = load_model_from_dir(checkpoint_dir, device=device, rank=ddp_rank)
max_seq_len = model_config_kwargs['seq_len']
model.train()

orig_model = model
# model = torch.compile(model, dynamic=True)
depth = model.config.n_layer
num_params = sum(p.numel() for p in model.parameters())
num_scaling_params = orig_model.num_scaling_params()
print(f"Number of parameters after compile: {num_params} (original: {num_scaling_params})")
num_flops_per_token = model.estimate_flops()
print(f"Estimated flops per token: {num_flops_per_token}")

# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
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

# Get tokenizer
tokenizer_dir = os.path.join(BASE_DIR, 'tokenizer')
tokenizer = get_tokenizer(tokenizer_dir)
token_bytes = get_token_bytes(tokenizer_dir, device=device)

# num iterations
examples_per_step = args.device_batch_size * ddp_world_size
print(f"Target examples per step: {args.target_examples_per_step}")
print(f"Device batch size: {args.device_batch_size}")
print(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}")
assert args.target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
grad_accum_steps = args.target_examples_per_step // examples_per_step
print(f"=> Setting grad accum steps: {grad_accum_steps}")

# Logging with tensorboard setup
if master_process:
    logging_tag = 'chat_sft'
    logging_dir = os.path.join(BASE_DIR, 'logs', logging_tag)
    writer = SummaryWriter(log_dir=logging_dir)


# define dataset
personality_filepath = os.path.join(BASE_DIR, 'identity_conversations.jsonl')
train_dataset = TaskMixture([
    ARC(subset="ARC-Easy", split="train"), # 2.3K rows
    ARC(subset="ARC-Challenge", split="train"), # 1.1K rows
    GSM8K(subset="main", split="train"), # 8K rows
    SmolTalk(split="train", size=10000), # 10K rows of smoltalk
    CustomJSON(filepath=personality_filepath, split="train"), # 1K rows of synthetic identity conversations
    SimpleSpelling(split="train", size=300), # 300 rows of Simple Spelling (e.g. spell the word 'apple')
    SpellingBee(split="train", size=300), # 300 rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
]) # 2.3K + 1.1K + 8K + 10K + 1K + 0.3K + 0.3K = 23K rows
val_dataset = TaskMixture([SmolTalk(split="test")]) # general conversations, 24K rows (though we don't actually use all of it)

train_loader = distributed_sft_data_loader(tokenizer, args.device_batch_size, max_seq_len, dataset=train_dataset, device=device)
val_loader = distributed_sft_data_loader(tokenizer, args.device_batch_size, max_seq_len, dataset=val_dataset, device=device)
x, y, epoch, _ = next(train_loader)

min_val_bpb = float('inf')
smooth_train_loss = 0
ema_beta = 0.9
total_training_time =  0
step = 1
last_step = False
progress = 0.0

def get_lr_multiplier(progress):
    return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

while True:
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        eval_steps = args.eval_tokens // (args.device_batch_size * max_seq_len * ddp_world_size)
        losses = []
        val_inputs, val_targets, *_ = next(val_loader)
        for _ in range(eval_steps):
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            losses.append(loss)
            val_inputs, val_targets, *_ = next(val_loader)
        val_loss = torch.stack(losses).mean()
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        if master_process:
            writer.add_scalar(f'{logging_tag}/val_loss', val_loss.item(), step)
        print(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
        model.train()
    
    if last_step or (args.task_eval_every > 0 and step % args.task_eval_every == 0):
        model.eval()
        metrics = {}
        with autocast_ctx:
            metrics['mmlu'] = eval_task('mmlu', model, tokenizer, max_problems=args.max_problems_per_task, k_shot=3) # accuracy@k
            metrics['arc-easy'] = eval_task('arc-easy', model, tokenizer, max_problems=args.max_problems_per_task, k_shot=3) # accuracy@k
        if master_process:
            for task_name, metric in metrics.items():
                writer.add_scalar(f'{logging_tag}/task_{task_name}_accuracy', metric, step)
        print(f'Step {step:05d} | Task Evaluation Metrics: {metrics}')
        model.train()
    
    if master_process and last_step:
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(BASE_DIR, 'chat_sft', output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            None,
            {
                'step': step,
                'model_config': model_config_kwargs,
                'user_config': user_config
            },
            ddp_rank
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
    if args.num_iterations == -1:
        progress = data_progress / args.num_epochs
    else:
        progress = step / args.num_iterations

    progress = min(progress, 1.0)

    lrm = get_lr_multiplier(progress)
    for opt in optimizers:
        for group in opt.param_groups:
            group['lr'] - group['initial_lr'] * lrm

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
    if step > 10:
        total_training_time += dt
    
    if master_process:
        writer.add_scalar(f'{logging_tag}/debiased_train_loss', debiased_smooth_loss, step)
    
    print(f"Step {step:05d} {progress * 100:.2f}% | loss: {debiased_smooth_loss} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | total time: {total_training_time/60:.2f}m")


    step += 1
    if step == args.num_iterations or (master_process and progress >= 1.0):
        last_step = True