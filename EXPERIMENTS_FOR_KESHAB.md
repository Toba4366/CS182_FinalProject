# Experiments to Run (Compute Plan for Keshab)

## Overview
This document outlines the experiment pipeline for FSM in-context learning. We need multiple runs (however many you see fit) for each configuration to compute error bars. Start with validating existing results, then expand to new configurations.

## File Organization
- **Experiment scripts**: `experiments/run_icl_*.py`
- **Run with**: `python -m experiments.script_name` (module import required)
- **Results**: Auto-saved to `checkpoints/` and `results/`
- **Multiple runs**: Use `--save_name` flag with suffixes like `_run1`, `_run2`, etc.

---

## Phase 1: Validate Existing Results (Priority 1)
**Goal**: Get 3+ runs of experiments we've already completed to establish error bars.

### 1.1 Large LSTM on 5s5a (Baseline)
```bash
# Run 3-5 times with different save names
python -m experiments.run_icl_lstm --epochs 16 --save_name lstm_5s5a_run1
python -m experiments.run_icl_lstm --epochs 16 --save_name lstm_5s5a_run2
python -m experiments.run_icl_lstm --epochs 16 --save_name lstm_5s5a_run3
# (Optional: run4, run5 for tighter error bars)
```
- **Model**: LSTM, d_model=256, 2 layers (~1M params)
- **Data**: 5 states, 5 actions (standard difficulty)
- **Epochs**: 32

### 1.2 Large Vanilla RNN on 5s5a (For Comparison)
```bash
python -m experiments.run_icl_vanilla_rnn --epochs 21 --save_name vanilla_rnn_5s5a_run1
python -m experiments.run_icl_vanilla_rnn --epochs 21 --save_name vanilla_rnn_5s5a_run2
python -m experiments.run_icl_vanilla_rnn --epochs 21 --save_name vanilla_rnn_5s5a_run3
```
- **Model**: Vanilla RNN, d_model=256, 2 layers
- **Expected**: Should perform worse than LSTM (we already know this)

---

## Phase 2: Fair Parameter Comparison (Priority 1)
**Goal**: Compare LSTM vs Transformer with matched parameters (~67K params, d_model=64).

### 2.1 Small LSTM on 5s5a
```bash
python -m experiments.run_icl_small_models --model lstm --d_model 64 --epochs 32 --save_name small_lstm_5s5a_run1
python -m experiments.run_icl_small_models --model lstm --d_model 64 --epochs 32 --save_name small_lstm_5s5a_run2
python -m experiments.run_icl_small_models --model lstm --d_model 64 --epochs 32 --save_name small_lstm_5s5a_run3
```
- **Model**: LSTM, d_model=64, 2 layers (~67K params)
- **Why**: Fair comparison with Keshab's transformer (also d_model=64)
- **Epochs**: 32 (less capacity = fewer epochs needed)
### 2.2 Small Vanilla RNN on 5s5a
```bash
python -m experiments.run_icl_small_models --model vanilla_rnn --d_model 64 --epochs 32 --save_name small_vanilla_5s5a_run1
python -m experiments.run_icl_small_models --model vanilla_rnn --d_model 64 --epochs 32 --save_name small_vanilla_5s5a_run2
python -m experiments.run_icl_small_models --model vanilla_rnn --d_model 64 --epochs 32 --save_name small_vanilla_5s5a_run3
```
- **Model**: Vanilla RNN, d_model=64, 2 layers
- **Why**: Check if small vanilla RNN can still work (probably not)

---

## Phase 3: Capacity vs Architecture Test (Priority 1)
**Goal**: Does vanilla RNN fail due to capacity or architecture?

### 3.1 Deep Vanilla RNN on 5s5a
```bash
python -m experiments.run_icl_deep_rnn --num_layers 8 --d_model 256 --epochs 32 --save_name deep_rnn_5s5a_run1
python -m experiments.run_icl_deep_rnn --num_layers 8 --d_model 256 --epochs 32 --save_name deep_rnn_5s5a_run2
python -m experiments.run_icl_deep_rnn --num_layers 8 --d_model 256 --epochs 32 --save_name deep_rnn_5s5a_run3
```
- **Model**: 8-layer Vanilla RNN, d_model=256 (~1M params, matches 2-layer LSTM)
- **Why**: If this works, vanilla RNN just needed more layers. If not, architecture matters.
- **Epochs**: 32 (same as large LSTM)

---

## Phase 4: Scaling to Harder FSMs (Priority 2)
**Goal**: Test if models can handle more complex FSMs (more states/actions).

### 4.1 Large LSTM on 5s8a (5 states, 8 actions)
```bash
python -m experiments.run_icl_5s8a --model lstm --epochs 48 --save_name lstm_5s8a_run1
python -m experiments.run_icl_5s8a --model lstm --epochs 48 --save_name lstm_5s8a_run2
python -m experiments.run_icl_5s8a --model lstm --epochs 48 --save_name lstm_5s8a_run3
```
- **Model**: LSTM, d_model=256, 2 layers
- **Data**: 5 states, 8 actions (harder - more actions per state)
- **Epochs**: 48 (32 + 16 for increased difficulty)
- **Note**: If this doesn't learn well, try curriculum learning (see Phase 6)

### 4.2 Small LSTM on 5s8a (If 4.1 succeeds)
```bash
# Only run if small LSTM learns 5s5a well
python -m experiments.run_icl_5s8a --model lstm --d_model 64 --epochs 64 --save_name small_lstm_5s8a_run1
python -m experiments.run_icl_5s8a --model lstm --d_model 64 --epochs 64 --save_name small_lstm_5s8a_run2
python -m experiments.run_icl_5s8a --model lstm --d_model 64 --epochs 64 --save_name small_lstm_5s8a_run3
```
- **Epochs**: 64 (32 + 32 for difficulty)
- **Conditional**: Only if 4.1 shows good learning

### 4.3 Large LSTM on 8s8a (8 states, 8 actions) - Only if 5s8a works
```bash
# Only run if 5s8a works well
python -m experiments.run_icl_8s8a --model lstm --epochs 100 --save_name lstm_8s8a_run1
python -m experiments.run_icl_8s8a --model lstm --epochs 100 --save_name lstm_8s8a_run2
python -m experiments.run_icl_8s8a --model lstm --epochs 100 --save_name lstm_8s8a_run3
```
- **Data**: 8 states, 8 actions (hardest configuration)
- **Epochs**: 100 (64 + 36 for increased difficulty)
- **Conditional**: Only proceed if 5s8a shows strong learning

---

## Phase 5: Linear Probing Analysis (Priority 2)
**Goal**: Compare representation quality between large vs small models.

### 5.1 Linear Probe on Large LSTM (5s5a)
```bash
# Use best checkpoint from Phase 1.1
python -m experiments.run_lstm_linear_probe \
    --checkpoint checkpoints/lstm_5s5a_run1_best.pt \
    --data_path data/icl_dataset.pt \
    --epochs 16 \
    --output_dir results/linear_probe_large_lstm_run1

# Repeat for run2, run3
python -m experiments.run_lstm_linear_probe \
    --checkpoint checkpoints/lstm_5s5a_run2_best.pt \
    --data_path data/icl_dataset.pt \
    --epochs 16 \
    --output_dir results/linear_probe_large_lstm_run2

python -m experiments.run_lstm_linear_probe \
    --checkpoint checkpoints/lstm_5s5a_run3_best.pt \
    --data_path data/icl_dataset.pt \
    --epochs 16 \
    --output_dir results/linear_probe_large_lstm_run3
```

### 5.2 Linear Probe on Small LSTM (5s5a)
```bash
# Use best checkpoint from Phase 2.1
python -m experiments.run_lstm_linear_probe \
    --checkpoint checkpoints/small_lstm_5s5a_run1_best.pt \
    --data_path data/icl_dataset.pt \
    --epochs 32 \
    --output_dir results/linear_probe_small_lstm_run1

# Repeat for run2, run3
python -m experiments.run_lstm_linear_probe \
    --checkpoint checkpoints/small_lstm_5s5a_run2_best.pt \
    --data_path data/icl_dataset.pt \
    --epochs 32 \
    --output_dir results/linear_probe_small_lstm_run2

python -m experiments.run_lstm_linear_probe \
    --checkpoint checkpoints/small_lstm_5s5a_run3_best.pt \
    --data_path data/icl_dataset.pt \
    --epochs 32 \
    --output_dir results/linear_probe_small_lstm_run3
```

### 5.3 Linear Probe on 5s8a Models (If they learn well... tune epochs as needed)
```bash
# Large LSTM 5s8a
python -m experiments.run_lstm_linear_probe \
    --checkpoint checkpoints/lstm_5s8a_run1_best.pt \
    --data_path data/icl_dataset_5s8a.pt \
    --epochs 16 \
    --output_dir results/linear_probe_large_lstm_5s8a_run1

# Small LSTM 5s8a (if trained)
python -m experiments.run_lstm_linear_probe \
    --checkpoint checkpoints/small_lstm_5s8a_run1_best.pt \
    --data_path data/icl_dataset_5s8a.pt \
    --epochs 16 \
    --output_dir results/linear_probe_small_lstm_5s8a_run1
```

**Key Question**: Does small model (64-dim) create linearly separable representations like large model (256-dim)?

---

## Phase 6: Curriculum Learning (Fallback Strategy)
**Goal**: If direct training fails, use curriculum learning.

### 6.1 Small LSTM Curriculum (If Phase 2.1 fails)
```bash
# Only if small LSTM doesn't learn 5s5a directly
python -m experiments.run_icl_lstm_curriculum \
    --dataset_type easy \
    --epochs 16 \
    --save_name small_lstm_curriculum_stage1

# Then continue training on standard
python -m experiments.run_icl_lstm_curriculum \
    --dataset_type standard \
    --epochs 16 \
    --checkpoint checkpoints/small_lstm_curriculum_stage1_best.pt \
    --save_name small_lstm_curriculum_stage2

# Then move up to hard (5s8a) curriculum

```

### 6.2 Large LSTM Curriculum for 5s8a (If Phase 4.1 struggles)
```bash
# Only if direct 5s8a training doesn't work well
# First train on 5s5a (already done in Phase 1)
# Then fine-tune on 5s8a
python -m experiments.run_icl_5s8a \
    --model lstm \
    --epochs 32 \
    --checkpoint checkpoints/lstm_5s5a_run1_best.pt \
    --save_name lstm_curriculum_5s5a_to_5s8a_run1
```

### 6.3 Large LSTM Curriculum for 5s8a (If Phase 4.3 struggles)
```bash
# Only if direct 8s8a training doesn't work well
# First train on 5s8a (already done in Phase 1)
# Then fine-tune on 8s8a
python -m experiments.run_icl_5s8a \
    --model lstm \
    --epochs 32 \
    --checkpoint checkpoints/lstm_5s8a_run1_best.pt \
    --save_name lstm_curriculum_5s8a_to_8s8a_run1
```
---

## Phase 7: Absorption State Analysis (Priority 3)
**Goal**: Test hypothesis that absorption states make learning easier.

### 7.1 Large LSTM with Absorption States
```bash
python -m experiments.run_icl_absorption --model lstm --epochs 32 --save_name lstm_absorption_run1
python -m experiments.run_icl_absorption --model lstm --epochs 32 --save_name lstm_absorption_run2
python -m experiments.run_icl_absorption --model lstm --epochs 32 --save_name lstm_absorption_run3
```
- **Expected**: Should learn faster/better than standard 5s5a (absorbing state = easier)

---

## Recommended Execution Order

1. **Start Here** (validate + establish baselines):
   - Phase 1.1: Large LSTM 5s5a (32 epochs × 3 runs)
   - Phase 2.1: Small LSTM 5s5a (16 epochs × 3 runs)
   - Phase 3.1: Deep RNN 5s5a (32 epochs × 3 runs)

2. **Linear Probing** (while other jobs run):
   - Phase 5.1: Probe large LSTM (16 epochs × 3 runs)
   - Phase 5.2: Probe small LSTM (16 epochs × 3 runs)

3. **Scaling Up** (only if baselines work):
   - Phase 4.1: Large LSTM 5s8a (48 epochs × 3 runs)
   - Phase 4.2: Small LSTM 5s8a (32 epochs × 3 runs) - conditional
   - Phase 4.3: Large LSTM 8s8a (64 epochs × 3 runs) - conditional

4. **Curriculum** (only if needed):
   - Phase 6.1: Small LSTM curriculum - if Phase 2.1 fails
   - Phase 6.2: Large LSTM 5s8a curriculum - if Phase 4.1 struggles

5. **Bonus** (if compute available):
   - Phase 1.2: Large vanilla RNN (for completeness)
   - Phase 2.2: Small vanilla RNN (probably fails)
   - Phase 7.1: Absorption states (interesting hypothesis)

---

## Expected Compute Requirements

**Per single run estimates** (rough):
- 5s5a, 32 epochs: ~2-4 hours on GPU
- 5s5a, 16 epochs: ~1-2 hours on GPU
- 5s8a, 48 epochs: ~4-6 hours on GPU
- 8s8a, 64 epochs: ~6-8 hours on GPU
- Linear probe, 16 epochs: ~1 hour on GPU

**Total for Priority 1+2** (minimum viable):
- Phase 1.1: 3 runs × 3 hours = 9 hours
- Phase 2.1: 3 runs × 1.5 hours = 4.5 hours
- Phase 3.1: 3 runs × 3 hours = 9 hours
- Phase 5.1-5.2: 6 runs × 1 hour = 6 hours
- **Total: ~28-30 hours GPU time**

---

## Data Files (Auto-generated)

Scripts automatically generate and cache datasets:
- `data/icl_dataset.pt` - Standard 5s5a (auto-created)
- `data/icl_dataset_5s8a.pt` - 5 states, 8 actions (auto-created)
- `data/icl_dataset_8s8a.pt` - 8 states, 8 actions (auto-created)
- `data/icl_dataset_absorption_5s5a.pt` - With absorption states (auto-created)

No manual dataset creation needed! Just run the scripts.

---

## Aggregating Results

After multiple runs, compute statistics:
```bash
python -m experiments.aggregate_results
```

This will search `results/` and `checkpoints/training_logs/` for matching experiment names and compute mean ± std for plotting with error bars.

---

## Key Hypotheses to Test

1. **Parameter Fairness**: Does small LSTM (64-dim, ~67K params) match transformer performance when parameter counts are equal?
2. **Capacity vs Architecture**: Does deep vanilla RNN (8 layers, ~1M params) match LSTM with same parameter budget?
3. **Linear Separability**: Do small models learn linearly separable representations like large models?
4. **Scaling**: Can models handle more complex FSMs (5s8a, 8s8a)?
5. **Curriculum**: Does curriculum learning help when direct training fails?

---

## Notes

- **16 is the magic number**: Base epochs for small models, increments for difficulty scaling
- **Always save with unique names**: Use `_run1`, `_run2`, `_run3` suffixes
- **Check results before proceeding**: Don't run 8s8a if 5s8a fails
- **Error bars are essential**: Need 5+ runs minimum for statistical validity
- **Module imports matter**: Always use `python -m experiments.script_name`
- **Watch for max_seq_len errors**: Already fixed, but verify datasets generate correctly

Good luck! 🚀
