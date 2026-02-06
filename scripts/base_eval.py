import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse

import torch

from utils.checkpoint import load_model_from_dir
from utils.common import get_base_dir, autodetect_device_type, compute_init, print0, compute_cleanup
from utils.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from bpe.tokenizer import get_tokenizer, get_token_bytes

from nanochat.sample_eval import get_response
from nanochat.loss_eval import evaluate_bpb
from nanochat.core_eval import evaluate_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model on train / val after pre-train / base-train')
    parser.add_argument('--device-type', type=str, default='', help='device type to use. cuda|mps|cpu. defulat: autodetect')
    parser.add_argument('--device-batch-size', type=int, default=32, help='per-device batch size')
    parser.add_argument('--eval-tokens', type=int, default=40*524288, help='tokens for eval task')
    parser.add_argument('--max-per-task', type=int, default=-1, help='max examples to use for each task. -1=disabled.')
    parser.add_argument('--model-tag', type=str, default='', help='model tag for loading model')
    # parser.add_argument('--step', type=int, default=None, help='optional step for loading model')
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

    BASE_DIR = get_base_dir()
    # load model
    checkpoint_dir = os.path.join(BASE_DIR, 'base_checkpoints', args.model_tag)
    model, model_config_kwargs = load_model_from_dir(checkpoint_dir, device=device, rank=ddp_rank)
    seq_len = model_config_kwargs['seq_len']
    model.eval()

    # load tokenizer
    tokenizer_dir = os.path.join(BASE_DIR, 'tokenizer')
    tokenizer = get_tokenizer(tokenizer_dir)
    token_bytes = get_token_bytes(tokenizer_dir, device)

    # bpb loss eval
    tokens_per_step = args.device_batch_size * seq_len * ddp_world_size
    assert args.eval_tokens % tokens_per_step == 0
    num_iterations = args.eval_tokens // tokens_per_step

    for split in ('train', 'val'):
        dataloader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, seq_len, split)
        input, target, *_ = next(dataloader)

        with autocast_ctx:
            bpb = evaluate_bpb(model, dataloader, num_iterations, token_bytes)
        print0(f'{split} bpb: {bpb:.6f}')
    
    # sample eval
    prompts = [
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "The opposite of hot is",
        "The planets of the solar system are:",
        "My favorite color is",
        "If 5*x + 3 = 13, then x is",
    ]
    if ddp_rank == 0:
        with autocast_ctx:
            for prompt in prompts:
                response = get_response(model, tokenizer, prompt, max_tokens=16)
                print0(f'Input prompt: {prompt} --> Response: {response}')
    
    # unconditioned samples from model as nanochat did
    if ddp_rank == 0:
        with autocast_ctx:
            for _ in range(8):
                response = get_response(model, tokenizer, "", max_tokens=128, temperature=1.0)
                print0(f'Unconditioned sample {_}: {response}')
    
    # core metric eval
    with autocast_ctx:
        results = evaluate_model(model, tokenizer, device, max_per_task=args.max_per_task)
        # print0(f" CORE metric: {results['core_metric']:.4f}")
    

    compute_cleanup()
