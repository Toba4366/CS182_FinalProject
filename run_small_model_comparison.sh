#!/bin/bash
# Complete comparison: small model (d_model=64) vs large model (d_model=256)
# For both training and linear probing

echo "========================================"
echo "Small Model (d_model=64) vs Large Model (d_model=256) Comparison"
echo "========================================"
echo ""

# Step 1: Train small LSTM (d_model=64) on 5s5a
echo "Step 1: Training small LSTM (d_model=64) on 5s5a..."
python -m experiments.run_icl_small_models \
    --model lstm \
    --d_model 64 \
    --epochs 50 \
    --save_name small_lstm_5s5a
echo "✓ Small LSTM trained"
echo ""

# Step 2: Linear probe on small LSTM
echo "Step 2: Linear probing small LSTM..."
python -m experiments.run_lstm_linear_probe \
    --checkpoint checkpoints/small_lstm_5s5a_best.pt \
    --data_path data/icl_dataset.pt \
    --epochs 20 \
    --output_dir results/linear_probe_small_lstm
echo "✓ Small LSTM linear probe complete"
echo ""

# Step 3: For comparison, re-run linear probe on existing large LSTM (d_model=256)
echo "Step 3: Linear probing large LSTM (d_model=256) for comparison..."
python -m experiments.run_lstm_linear_probe \
    --checkpoint checkpoints/lstm_best.pt \
    --data_path data/icl_dataset.pt \
    --epochs 20 \
    --output_dir results/linear_probe_large_lstm
echo "✓ Large LSTM linear probe complete"
echo ""

echo "========================================"
echo "Comparison Complete!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - Small LSTM training: checkpoints/small_lstm_5s5a_best.pt"
echo "  - Small LSTM probe: results/linear_probe_small_lstm/"
echo "  - Large LSTM probe: results/linear_probe_large_lstm/"
echo ""
echo "Compare the linear probe accuracies to see if model size affects"
echo "the quality of learned representations for linear separability."
