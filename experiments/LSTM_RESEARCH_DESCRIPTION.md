# LSTM Architecture Exploration for In-Context Learning of Finite State Moore Machines

## Overview

We explored LSTM architectures as an alternative to transformers for in-context learning (ICL) of Finite State Moore Machines (FSMs). Similar to our transformer experiments, we tested various LSTM configurations to understand their capabilities in learning FSM dynamics from state-action demonstrations.

## LSTM Architecture Exploration

We tested a variety of LSTM architectures with different hidden dimensions (`d_model`) and number of layers. The architectures tested are shown in the following table:

| Hidden dimension | Number of layers | Parameters |
|-----------------|------------------|------------|
| 32 | 1 | 9,024 |
| 32 | 2 | 17,472 |
| 64 | 1 | 34,432 |
| 64 | 2 | 67,712 |
| 128 | 1 | 134,400 |
| 128 | 2 | 266,496 |
| 256 | 1 | 530,944 |
| 256 | 2 | 1,057,280 |

### Hyperparameter Search

For LSTM models, we conducted a hyperparameter sweep over:
- **Learning rates**: [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
- **Batch sizes**: [16, 24, 32]
- **Weight decays**: [0.0, 1e-5, 1e-4]

All models used the **AdamW optimizer**.

### Best Hyperparameter Settings

**Smaller LSTM (d_model=64, num_layers=2):**
The best hyperparameter configuration found was:
- **Learning rate**: 5e-3
- **Weight decay**: 0.0
- **Batch size**: 24

**Larger LSTM (d_model=256, num_layers=1):**
The best hyperparameter configuration found was:
- **Learning rate**: 2e-3
- **Weight decay**: 0.0
- **Batch size**: 24

### Best Performing Architectures

**Closest to Best Transformer:**
The LSTM model with **d_model = 64** and **num_layers = 2** (approximately **67,712 parameters**) was the closest match to our best performing transformer architecture (which had ~52,424 parameters with d_model=64, num_heads=4, d_ff=64, num_layers=2). This LSTM configuration achieved competitive performance while maintaining a similar parameter count.

**Best Overall LSTM Performance:**
The LSTM model with **d_model = 256** and **num_layers = 1** (approximately **530,944 parameters**, nearly 500k) achieved the best performance among all LSTM configurations, reaching near-perfect accuracy. However, this came at the cost of significantly more parameters compared to the transformer baseline.

### Training Methodology

**Direct Training:**
Unlike transformers, we found that **curriculum learning was not very helpful for LSTMs**. Instead, we trained the LSTM models directly on the **5 states, 5 actions (5s5a)** dataset without the progressive curriculum approach. This direct training strategy proved more effective for LSTM architectures.

### Comparison with Transformers

While LSTMs can achieve strong performance on this ICL task, they generally require more parameters than transformers to reach similar accuracy levels. The best performing LSTM (d_model=256, num_layers=1) required approximately 10x more parameters than our best transformer (d_model=64, num_heads=4, d_ff=64, num_layers=2) to achieve near-perfect accuracy.

## Extrapolation Testing

We tested the best LSTM models on test queries of longer lengths to assess their ability to generalize beyond the training distribution. The models were evaluated on query lengths of **200, 300, 400, 500, and 600** to understand their scaling behavior.

