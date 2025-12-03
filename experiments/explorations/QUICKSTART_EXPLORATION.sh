#!/bin/bash
# Quick Start: Run a single exploratory experiment
# Usage: ./QUICKSTART_EXPLORATION.sh

echo "════════════════════════════════════════════════════════════════"
echo "  RNN EXPLORATION - QUICK START"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "This script will run ONE quick experiment to verify setup:"
echo "  → GRU baseline (fastest, ~15-20 min)"
echo ""
echo "For full exploration, use: python experiments/explorations/run_all_exploration.py"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""

# Set Python path
export PYTHONPATH="${PWD}/../..:$PYTHONPATH"

# Run GRU baseline experiment
echo "🚀 Starting GRU baseline experiment..."
echo ""

cd ../..
python experiments/explorations/run_gru_experiment.py \
    --d-model 256 \
    --num-layers 2 \
    --epochs 20 \
    --batch-size 8 \
    --experiment-name gru_baseline

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ QUICK START COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Check results in: experiments/explorations/results/"
echo "  2. Run full suite: python experiments/explorations/run_all_exploration.py --mode all"
echo "  3. Analyze in notebook: results/architecture_comparison/rnn_exploration_analysis.ipynb"
echo ""
