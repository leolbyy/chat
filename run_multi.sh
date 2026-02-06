export CHAT_BASE_DIR="$HOME/project_chat"

uv sync --extra=gpu
source .venv/bin/activate

cp -r $HOME/tokenizer $HOME/project_chat/

# torchrun --standalone --nproc_per_node=2 -m scripts.data_downloader --type=train -n 16 &
# torchrun --standalone --nproc_per_node=2 -m scripts.data_downloader --type=eval
# torchrun --standalone --nproc_per_node=2 -m scripts.data_downloader --type=personality
# torchrun --standalone --nproc_per_node=2 -m scripts.data_downloader --type=spell


torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --tokens-per-step=65536 \
    --eval-every=25 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=25 \
    --num-iterations=100
torchrun --standalone --nproc_per_node=2 -m scripts.base_eval \
    --eval-tokens=524288 \
    --max-per-task=16 \
    --model-tag=d6

# midtraining (~10 minutes on my MacBook Pro M3 Max)
torchrun --standalone --nproc_per_node=2 -m scripts.mid_train \
    --device-batch-size=32 \
    --tokens-per-step=65536 \
    --eval-every=25 \
    --eval-tokens=524288 \
    --num-iterations=100 \
    --model-tag=d6

torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft \
    --device-batch-size=32 \
    --model-source=mid \
    --model-tag=d6 \
    --eval-every=25 \
    --eval-tokens=524288 \
    --task-eval-every=25 \
    --max-problems-per-task=16 \
    --target-examples-per-step=64
