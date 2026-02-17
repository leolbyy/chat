CURRENT_HOSTNAME=$(hostname)

if [ -f /etc/hosts ]; then
    cp /etc/hosts /etc/hosts.bak
fi

cat > /etc/hosts <<EOF
127.0.0.1   localhost
127.0.0.1   ${CURRENT_HOSTNAME}
::1         localhost
::1         ${CURRENT_HOSTNAME}
EOF

source /etc/network_turbo

export CHAT_BASE_DIR="/root/autodl-tmp/chat"

if ! command -v uv &> /dev/null; then
    command -v curl &> /dev/null || (apt-get update && apt-get install -y curl)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

uv sync --extra=gpu
source .venv/bin/activate

python -m scripts.data_downloader --type=train -n 370 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.data_downloader --type=eval
# python -m scripts.data_downloader --type=personality
python -m scripts.data_downloader --type=spell

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

torchrun --standalone --nproc_per_node=2 -m scripts.base_train \
    --depth=26 \
    --target-param-data-ratio=8.25 \
    --device-batch-size=16 \
    --sample-kvcache-every=2000 \
    --save-every=2000

torchrun --standalone --nproc_per_node=2 -m scripts.base_eval \
    --model-tag=d26

# midtraining (~10 minutes on my MacBook Pro M3 Max)
torchrun --standalone --nproc_per_node=2 -m scripts.mid_train \
    --device-batch-size=16 \
    --model-tag=d26

torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft \
    --device-batch-size=16
