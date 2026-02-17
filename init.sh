#!/bin/bash
set -e

echo "=== Initializing ASL-LSTM Environment ==="

# 1. Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python: $python_version"

# 2. Sync dependencies with UV
if command -v uv &> /dev/null; then
    echo "Syncing dependencies with uv..."
    uv sync
else
    echo "Error: uv is not installed. Please install uv first."
    exit 1
fi

# 3. Create necessary directories if they don't exist
mkdir -p logs
mkdir -p src/checkpoints
mkdir -p dataset/processed

# 4. Check for raw data (optional warning)
if [ ! -d "dataset/raw" ]; then
    echo "Warning: dataset/raw directory not found. Please ensure WLASL data is placed there."
fi

echo "=== Initialization Complete ==="
echo "Run 'uv run python src/test/gpu_cuda_check.py' to verify GPU availability."
