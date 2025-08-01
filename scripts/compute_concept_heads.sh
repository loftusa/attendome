#!/bin/bash

set -e  # Exit on any error
cd ..

# Model names to process
MODELS=(
    "EleutherAI/gpt-neox-20b"
    "EleutherAI/gpt-j-6b"   
    "meta-llama/Llama-2-7b-hf"
    "meta-llama/Llama-2-13b-hf"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
)

# Loop through each model
for model in "${MODELS[@]}"; do
    echo "Processing model: $model"

    python ./compute_attention_scores.py --model "$model" --bsz 4

done