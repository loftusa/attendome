#!/bin/bash

set -e  # Exit on any error
cd ..

# Model names to process
MODELS=(
#    "mistral-7b"
#    "mistral-24b" 
    "gpt-j-6b"
    "gpt-neox-20b"
    "llama-2-7b"
    "llama-2-13b"
)

# Loop through each model
for model in "${MODELS[@]}"; do
    echo "Processing model: $model"

    python ./experiments/_01_compute_rdm.py --model_name "$model"

done