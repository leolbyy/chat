uv sync --extra=gpu
source .venv/bin/activate

python -m scripts.base_train \
    --depth=6 \
    --head-dim=128 \
    --max-seq-len=1024 \
    --device-batch-size=32 \
    --tokens-per-step=524288 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --sample-kvcache-every=500 \
    --save-every=1000 \
    --num-iterations=5000
python -m scripts.base_eval \
    --eval-tokens=524288 \
    --max-per-task=16 \
    --model-tag=d6

# # midtraining (~10 minutes on my MacBook Pro M3 Max)
# python -m scripts.mid_train \
#     --device-batch-size=32 \
#     --tokens-per-step=16384 \
#     --eval-every=25 \
#     --eval-tokens=524288 \
#     --num-iterations=100 \
#     --model-tag=d6

# python -m scripts.chat_sft \
#     --device-batch-size=32 \
#     --model-source=mid \
#     --model-tag=d6 \
#     --eval-every=25 \
#     --eval-tokens=524288 \
#     --task-eval-every=25 \
#     --max-problems-per-task=16