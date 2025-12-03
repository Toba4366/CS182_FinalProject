# RNN Exploration Experiments

This directory contains experiments exploring vanilla RNN improvements for In-Context Learning (ICL) on Finite State Machines.

## Research Questions

1. **Capacity Hypothesis**: Does increasing hidden dimension help vanilla RNNs?
2. **Depth Hypothesis**: Can deeper RNNs overcome shallow RNN limitations?
3. **Architecture Spectrum**: Where does GRU fall between RNN and LSTM?

## Experiments

### 1. Capacity Tests (`explore_rnn_capacity.py`)

Test RNN performance with varying hidden dimensions:
- **d_model = 256** (baseline): ~200K parameters
- **d_model = 512**: ~800K parameters (similar to LSTM)
- **d_model = 1024**: ~3M parameters (4x LSTM)

**Run:**
```bash
# Small
python experiments/explorations/explore_rnn_capacity.py --d-model 256 --experiment-name rnn_d256_baseline

# Medium
python experiments/explorations/explore_rnn_capacity.py --d-model 512 --experiment-name rnn_d512

# Large
python experiments/explorations/explore_rnn_capacity.py --d-model 1024 --experiment-name rnn_d1024
```

### 2. Depth Tests (`explore_rnn_capacity.py`)

Test RNN performance with varying depths:
- **2 layers** (baseline): Standard depth
- **5 layers**: Medium deep
- **16 layers**: Very deep (ResNet-style)

**Run:**
```bash
# Shallow
python experiments/explorations/explore_rnn_capacity.py --num-layers 2 --experiment-name rnn_l2_baseline

# Medium
python experiments/explorations/explore_rnn_capacity.py --num-layers 5 --experiment-name rnn_l5

# Deep
python experiments/explorations/explore_rnn_capacity.py --num-layers 16 --experiment-name rnn_l16
```

### 3. GRU Comparison (`run_gru_experiment.py`)

Fill the gap between vanilla RNN and LSTM with GRU:

**Run:**
```bash
python experiments/explorations/run_gru_experiment.py --experiment-name gru_baseline
```

### 4. Run All Experiments (`run_all_exploration.py`)

Orchestrate all experiments:

```bash
# Run everything
python experiments/explorations/run_all_exploration.py --mode all

# Or run specific categories
python experiments/explorations/run_all_exploration.py --mode capacity
python experiments/explorations/run_all_exploration.py --mode depth
python experiments/explorations/run_all_exploration.py --mode gru
```

## Results Structure

All results saved to `experiments/explorations/results/`:
```
experiments/explorations/results/
├── rnn_d256_baseline_TIMESTAMP.pt
├── rnn_d256_baseline_TIMESTAMP_metrics.json
├── rnn_d512_TIMESTAMP.pt
├── rnn_d512_TIMESTAMP_metrics.json
├── rnn_d1024_TIMESTAMP.pt
├── rnn_d1024_TIMESTAMP_metrics.json
├── rnn_l2_baseline_TIMESTAMP.pt
├── rnn_l5_TIMESTAMP.pt
├── rnn_l16_TIMESTAMP.pt
├── gru_baseline_TIMESTAMP.pt
└── gru_baseline_TIMESTAMP_metrics.json
```

## Analysis Notebook

Comprehensive analysis in `results/architecture_comparison/rnn_exploration_analysis.ipynb`:

- **Capacity vs Performance**: Hidden dimension analysis
- **Depth vs Performance**: Layer count analysis  
- **Architecture Spectrum**: RNN → GRU → LSTM comparison
- **Training Curves**: Convergence analysis
- **Key Insights**: What matters for ICL?

## Expected Outcomes

### Hypothesis Predictions

**If capacity is key:**
- Large RNN (1024d) should match/exceed LSTM (256d)
- Parameter count should correlate with performance

**If depth is key:**
- Very deep RNN (16L) should approach LSTM performance
- Gradients might still vanish → limited gains

**If gating is key:**
- GRU should significantly outperform vanilla RNN
- GRU should approach LSTM performance
- Capacity/depth won't help vanilla RNN much

## Connection to Main Paper

These explorations help understand:
1. **Why LSTM succeeds**: Gating vs capacity vs depth
2. **Architecture design**: What factors enable ICL?
3. **Practical recommendations**: When to use which architecture

## Notes

- All experiments use same hyperparameters (lr=1e-3, batch_size=8)
- Training on same dataset as main experiments
- `experiments/explorations/` directory in `.gitignore` (experiments only, not for paper)
- Notebook provides comparison with published baseline results
