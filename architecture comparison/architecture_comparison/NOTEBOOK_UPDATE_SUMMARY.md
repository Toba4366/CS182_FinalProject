# Notebook Update Summary

## Date: November 25, 2025

## What Was Updated

The `icl_architecture_analysis.ipynb` notebook has been completely updated with **REAL experimental data** from all 6 completed experiments.

## Key Changes

### 1. Real Data Loading (Cell 3)
- ✅ Now loads actual training history from JSON files in `checkpoints/training_logs/`
- ✅ Loads all 6 experiments: LSTM/RNN × Direct/Curriculum Stage 1/Curriculum Stage 2
- ✅ Contains complete epoch-by-epoch training/validation losses and accuracies

### 2. Updated Results Summary (Cell 1)
- LSTM Direct: **99.85%** (was placeholder)
- LSTM Curriculum Stage 2: **98.82%** (was placeholder)
- Vanilla RNN Direct: **53.66%** (was placeholder)
- Vanilla RNN Curriculum Stage 2: **20.73%** (was placeholder - shows SEVERE negative transfer!)

### 3. Real Training Curves (Cells 11-12)
- ✅ **NEW:** Actual training curves plotted from real experimental data
- ✅ Shows validation accuracy curves for both architectures
- ✅ Shows loss curves (log scale) for convergence analysis
- ✅ Additional detailed analysis: curriculum stage 1 vs stage 2 comparison
- ✅ Training stability analysis (std dev of last 5 epochs)
- ✅ Convergence speed analysis (epochs to 90% of final accuracy)

### 4. Updated Analysis (Cell 14)
- ✅ All findings based on **REAL experimental results**
- ✅ Confirmed: LSTM achieves ~99% with or without curriculum
- ✅ **CRITICAL FINDING:** Vanilla RNN shows -33 percentage point drop with curriculum!
- ✅ Evidence of catastrophic forgetting in Vanilla RNN
- ✅ Direct training is BETTER than curriculum for both architectures

### 5. Data Export (Cell 15)
- ✅ Saves consolidated `training_curves_data.json` for paper figures
- ✅ Contains all training history in clean format for LaTeX/plotting

## Experimental Results Summary

| Architecture | Direct | Curriculum S1 (Simple) | Curriculum S2 (Complex) |
|--------------|--------|------------------------|-------------------------|
| **LSTM** | 99.85% | 99.68% | 98.82% |
| **Vanilla RNN** | 53.66% | 41.29% | **20.73%** ⚠️ |

## Key Research Findings

1. **Curriculum Learning NOT Beneficial**
   - LSTM: -1.03 percentage points (direct is slightly better)
   - Vanilla RNN: -32.93 percentage points (SEVERE negative transfer!)

2. **Architecture Matters**
   - LSTM: Near-perfect accuracy (~99%)
   - Vanilla RNN: Poor performance (<55%)
   - Gap: ~46 percentage points

3. **Catastrophic Forgetting**
   - Vanilla RNN drops from 41.29% (simple FSM) → 20.73% (complex FSM)
   - Cannot retain knowledge from stage 1
   - Direct training is 33 percentage points BETTER than curriculum!

4. **Convergence Speed**
   - LSTM: Fast (~16 epochs), stable
   - Vanilla RNN: Slow (~20 epochs), unstable, low accuracy

## Next Steps for Paper

✅ Use the real training curves in figures
✅ Reference `training_curves_data.json` for plotting
✅ Update paper text with actual percentages
✅ Emphasize negative transfer finding (novel contribution!)
✅ Recommend direct training over curriculum for FSM-ICL tasks

## Files Generated

- `training_curves_real.png` - Real training/loss curves
- `training_curves_detailed.png` - Detailed analysis with both stages
- `training_curves_data.json` - Raw data for paper figures
- All previous visualizations updated with real data
