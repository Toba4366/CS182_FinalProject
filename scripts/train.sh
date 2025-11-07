#!/bin/bash
# Training script for Moore machine ICL experiments

set -e  # Exit on error

# Default values
CONFIG_FILE="configs/base_config.yaml"
OUTPUT_DIR="results/experiment_$(date +%Y%m%d_%H%M%S)"
GPU_ID=""
USE_WANDB=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --wandb)
            USE_WANDB=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -c, --config FILE    Configuration file (default: configs/base_config.yaml)"
            echo "  -o, --output DIR     Output directory (default: results/experiment_TIMESTAMP)"
            echo "  -g, --gpu ID         GPU device ID to use"
            echo "      --wandb          Enable Weights & Biases logging"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "=== Moore Machine ICL Training ==="
echo "Config file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "GPU device: ${GPU_ID:-auto-detect}"
echo "W&B logging: $USE_WANDB"
echo

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set GPU environment variable if specified
if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_ID
fi

# Run experiment
echo "Starting training..."
python experiments/run_experiment.py \
    --config "$CONFIG_FILE" \
    --output "$OUTPUT_DIR" \
    ${GPU_ID:+--gpu $GPU_ID}

echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"

# Generate summary plots if training succeeded
echo "Generating result plots..."
python scripts/plot_results.py "$OUTPUT_DIR" || echo "Warning: Could not generate plots"

echo "=== Training Complete ==="