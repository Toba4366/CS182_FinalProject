# In-Context Learning of Finite State Moore Machines

## Goal

Our goal is to investigate whether transformer and RNN/LSTM-style models can perform in-context learning (ICL) of Finite State Moore Machines (FSMs) from seeing state-action pairs in demonstrations and then predicting next states based on a starting state and a list of actions.

## Dataset Generation

We generated approximately 10,000 samples randomly for FSMs with two configurations:
- **3 states, 3 actions** (3s3a)
- **5 states, 5 actions** (5s5a)

The dataset is split as follows:
- **6,000 samples** for training
- **2,000 samples** for validation
- **2,000 samples** for testing

## Perfect Solver Baseline

We created a perfect (deterministic) solver and conducted tests to determine ideal demo lengths such that the percentage of next state predictions would achieve at least **99.5% accuracy**. We randomly introduce a variation of **±20%** on the demo lengths to add variability to the training data.

### Observed Ideal Demo Lengths

- **3 states, 3 actions**: 3 demos with approximately **length 30** consistently achieved 99.5% accuracy
- **5 states, 5 actions**: 3 demos with approximately **length 90** consistently achieved 99.5% accuracy

Note: These demo lengths are randomly variable (±20%) to add diversity to the training examples.

## Transformer Architecture Exploration

We tested a variety of transformer architectures with **2 layers** (see rationale below). The architectures tested are shown in the following table:

| Transformer encoding dimension | num_heads | Feed forward network length | Parameters |
|-------------------------------|-----------|----------------------------|------------|
| 32 | 2 | 64 | 18,088 |
| 32 | 4 | 64 | 18,088 |
| 64 | 4 | 64 | 52,424 |
| 64 | 4 | 128 | 68,936 |
| 64 | 8 | 128 | 68,936 |
| 128 | 8 | 256 | 268,936 |
| 256 | 8 | 512 | 1,062,152 |

### Architecture Selection Rationale

We note that **3 transformer layers** should easily be able to achieve n-gram matching abilities for this task. To challenge the architecture and test its limits, we constrained all models to use **2 layers** to see if they could still achieve high performance on this task.

### Positional Embeddings

We use **Rotary Position Embedding (RoPE)** for positional encoding. Through experimentation, we observed that RoPE worked much better than absolute positional embeddings for this task, likely due to its ability to better capture relative positional relationships in the sequence.

### Hyperparameter Search

For transformer models, we conducted a hyperparameter sweep over:
- **Learning rates**: [1e-4, 5e-4, 1e-3, 3e-3, 1e-2]
- **Batch sizes**: [16, 24, 32]
- **Weight decays**: [0.0, 1e-5, 1e-4]

All models used the **AdamW optimizer**.

### Best Hyperparameter Setting

The best hyperparameter configuration found was:
- **Learning rate**: 1e-3
- **Weight decay**: 0.0
- **Batch size**: 24

### Best Performing Architecture

The smallest model that we found which trained consistently was:
- **d_model = 64**
- **num_heads = 4**
- **d_ff = 64**
- **num_layers = 2**
- **Total parameters: ~51,000**

## Training Methodology

### Attention Mechanism

We add a **causal mask** to the self-attention mechanism to ensure that each position in the sequence can only attend to previous positions and itself. This prevents the model from accessing future information during training, which is crucial for autoregressive prediction tasks.

### Training Procedure

We use **teacher forcing** during training, where the model receives the ground truth tokens as input at each time step, even if it made a prediction error in the previous step. This allows for more stable and efficient training.

### Loss Computation

The loss is computed only on the output positions corresponding to **states in the query** (i.e., the positions where we need to predict the next state given an action). All other positions (demos, actions, special tokens) are masked out from the loss calculation. This focuses the model's learning on the core task of state prediction.

### Evaluation Methodology

All test results report **next state prediction accuracy** - that is, the accuracy of predicting each individual state given the ground truth previous state and action. This evaluation does **not** measure the model's ability to predict entire sequences with compounding errors, where a single incorrect prediction would affect all subsequent predictions. Instead, each state prediction is evaluated independently using teacher forcing, providing a clearer picture of the model's core next-state prediction capability.

## Curriculum Learning

We observed that a **curriculum learning approach** worked well for transformers:
1. **Stage 1**: Train on 3 states, 3 actions (3s3a) to near-perfect accuracy
2. **Stage 2**: Use the same model (continuing training) on 5 states, 5 actions (5s5a)

This progressive training strategy allowed the model to first learn simpler patterns before tackling more complex state-action spaces.

## Extrapolation Testing

We further tested the model's next state prediction capabilities on test queries of longer lengths:
- **200, 300, 400, 500, and 600** query lengths

This tests the model's ability to generalize to longer sequences beyond the training distribution.

## Statistical Reporting

All reported results are based on multiple runs (5 runs) to ensure statistical robustness. In all line graphs:
- **Shaded regions** represent **1 standard deviation** from the mean across runs
- **Error bars** represent **1 standard deviation** from the mean across runs

This provides a clear visualization of the variability and consistency of our results.

