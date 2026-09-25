#!/usr/bin/env bash
# One-time server setup for Thunder Compute (or any Ubuntu GPU instance).
# Run once after SSH into the instance.
#
# Usage:
#   bash scripts/setup_server.sh

set -euo pipefail

echo "=== Installing system deps ==="
apt-get update -qq && apt-get install -y -qq tmux git curl

echo "=== Installing Python deps ==="
pip install --upgrade pip -q
pip install -q \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    transformers==4.40.0 \
    pyarrow \
    pandas \
    scikit-learn \
    tqdm \
    rich \
    requests \
    beautifulsoup4

echo "=== Installing package ==="
pip install -e . -q

echo "=== Verifying GPU ==="
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Upload data:  scp -r data/processed user@host:~/pharmviginet/data/"
echo "  2. Start tmux:   tmux new -s train"
echo "  3. Quick test:   bash scripts/run_pipeline.sh --sample 50000 --epochs 1"
echo "  4. Full train:   bash scripts/run_pipeline.sh --sample 500000"
echo "  5. Detach tmux:  Ctrl+B then D"
echo "  6. Reattach:     tmux attach -t train"
