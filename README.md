# CS 182 Final Project: In-Context Learning of Finite Automata

A comprehensive study of **In-Context Learning (ICL)** capabilities across different neural architectures when learning deterministic finite automata state transitions. This project compares how Transformers, LSTMs, and Vanilla RNNs perform on sequential pattern learning tasks.

**Team Members:** Trenton O'Bannon, Yuri Lee, Keshab Agarwal, Evan Davis

---

## 🎯 Project Overview

### Motivation

**In-Context Learning** is the remarkable ability of neural models to learn new patterns from a few examples provided in their input context, without updating their parameters. While this capability has been extensively studied in large language models (Transformers), little research exists on how different architectures compare on structured sequential learning tasks.

**Deterministic Finite Automata (DFAs)** provide an ideal testbed because they represent:
- **Deterministic state transitions**: Each state-action pair maps to exactly one next state
- **Learnable patterns**: Complex enough for meaningful comparison, simple enough to analyze
- **Mathematical structure**: Well-defined properties for validating model correctness
- **Scalable complexity**: Configurable number of states and actions for progressive difficulty

### Research Questions

1. **Architecture Comparison**: How do different neural architectures (Transformer, LSTM, Vanilla RNN) perform on ICL for finite state machines?
2. **Sequence Modeling**: Which architectural inductive biases are most effective for learning finite state machine patterns?
3. **ICL Mechanisms**: What makes certain architectures better at in-context learning for sequential decision tasks?
4. **DFA Properties**: How do different DFA characteristics (absorbing states, connectivity) affect learnability?

---

## 🏗️ Project Architecture

### Repository Structure

```
CS182_FinalProject/
├── src/                              # Source code
│   ├── models/                       # Neural architecture implementations
│   │   ├── moore_transformer.py      # Transformer with RoPE and causal attention
│   │   ├── moore_vanilla_rnn.py      # Vanilla RNN with tanh/relu activations
│   │   └── moore_lstm.py             # LSTM with optional bidirectionality
│   ├── training/                     # Model-specific trainers
│   │   ├── icl_trainer.py            # Transformer ICL trainer
│   │   ├── vanilla_rnn_trainer.py    # Vanilla RNN ICL trainer  
│   │   └── lstm_trainer.py           # LSTM ICL trainer
│   ├── datasets/                     # Data generation and loading
│   │   └── moore_dataset.py          # DFA trajectory dataset for ICL
│   ├── fsm/                          # Finite state machine utilities
│   │   ├── generator.py              # DFA generation with deterministic transitions
│   │   └── trajectory_sampler.py     # FSM trajectory sampling for ICL sequences
│   └── utils/                        # Utility functions
├── experiments/                      # Training scripts
│   ├── run_icl_transformer.py        # Train Transformer model
│   ├── run_icl_vanilla_rnn.py        # Train Vanilla RNN model  
│   ├── run_icl_lstm.py               # Train LSTM model
│   └── run_icl_lstm_curriculum.py    # Curriculum learning for LSTM
├── tests/                            # Comprehensive test suite
│   ├── test_moore_models.py          # Model functionality tests
│   └── test_icl_trainer.py           # Trainer functionality tests
├── data/                             # Generated datasets
│   └── icl_dataset.pt                # Pre-generated DFA trajectories
├── checkpoints/                      # Saved model checkpoints
└── results/                          # Training metrics and plots
```

### Key Design Principles

1. **Model-Agnostic ICL Interface**: All models implement the same `forward(input_ids, targets, unknown_mask)` signature for consistent evaluation
2. **Individual Trainers**: Each architecture has a dedicated trainer optimized for its specific requirements
3. **Comprehensive Testing**: Full test coverage ensures model correctness and interface consistency  
4. **DFA Focus**: All components specialized for deterministic finite automata learning tasks

---

## 🧠 Model Architectures

### 1. Moore Transformer (`moore_transformer.py`)
- **Decoder-only architecture** with causal attention
- **Rotary Position Embedding (RoPE)** for better sequence understanding  
- **Multi-head attention** with configurable heads and layers
- **Specialized for ICL** with unknown state masking
- **Parameters**: ~52K (d=64, 2 layers, 4 heads)

### 2. Moore Vanilla RNN (`moore_vanilla_rnn.py`)
- **Elman network** with tanh/relu activations
- **Multi-layer support** with dropout between layers
- **Step-by-step processing** maintaining hidden states across timesteps
- **Minimal architecture** serving as baseline for comparison
- **Parameters**: ~68K (d=128, 2 layers)

### 3. Moore LSTM (`moore_lstm.py`)
- **Long Short-Term Memory** with forget/input/output gates
- **Bidirectional support** for enhanced context modeling
- **Proper initialization** with orthogonal hidden-to-hidden weights
- **Gradient-friendly** design for longer sequences
- **Parameters**: ~531K (d=256, 1 layer) for high accuracy

---

## 📊 In-Context Learning Setup

### Sequence Format

Each training sequence follows the ICL paradigm:

```
[Demo 1] S₀, A₀ → S₁, A₁ → S₂, ... <eos>
[Demo 2] S₀, A₀ → S₁, A₁ → S₂, ... <eos>
[Demo 3] S₀, A₀ → S₁, A₁ → S₂, ... <eos>
<query> S₀, A₀ → [?], A₁ → [?], A₂ → [?] <eos>
```

Where:
- **Demo Examples**: Known state transitions the model learns from
- **Query Segment**: Unknown states (`[?]`) the model must predict
- **Loss Masking**: Only unknown positions contribute to loss

### Vocabulary Structure

- **State Tokens**: `0 ... num_states-1`
- **Action Tokens**: `num_states ... num_states + max_actions - 1`
- **Special Tokens**:
  - `<eos>`: Separates demo from query segments
  - `<query>`: Marks start of query portion
  - `<pad>`: Padding for batch processing

### Dataset Generation

The `MooreICLDataset` creates sequences by:
1. **Generating diverse DFAs** with varying complexity using the deterministic generator
2. **Sampling demonstration trajectories** showing valid state transitions
3. **Creating query segments** with unknown states to predict
4. **Applying loss masking** to focus learning on unknowns

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install torch numpy
```

### 2. Generate dataset

```bash
python -c "from src.datasets.moore_dataset import load_or_create_icl_samples, ICLDatasetConfig; load_or_create_icl_samples(ICLDatasetConfig())"
```

### 3. Train models

```bash
# Train Transformer with curriculum learning
python experiments/run_icl_transformer.py --batch-size 24 --num-layers 2

# Train LSTM directly
python experiments/run_icl_lstm.py --epochs 20 --d-model 256 --num-layers 1

# Train Vanilla RNN
python experiments/run_icl_vanilla_rnn.py --epochs 10 --d-model 256 --activation tanh
```

### 4. Run tests

```bash
# Run all tests
python -m pytest tests/ -v

# Test specific components
python -m pytest tests/test_moore_models.py -v
```

---

## 🔬 Key Features

### DFA Generation (`src/fsm/generator.py`)

The generator creates **true deterministic finite automata** with:
- **Deterministic Transitions**: Every state has exactly `action_count` outgoing transitions
- **Configurable Complexity**: 
  - `num_states`: Number of states in the automaton (default: 5)
  - `min_actions`/`max_actions`: Range of actions per state (default: 3-8)
- **Absorbing State Support**: Optional generation of DFAs with absorbing states
- **Connectivity**: Ensures all states are reachable from initial state

### Curriculum Learning

Transformers require curriculum learning to solve complex DFAs:

```python
# Stage 1: Simple DFAs (3 states, 3 actions)
# Stage 2: Complex DFAs (5 states, 4-5 actions)

stages = [
    CurriculumStage(num_states=3, max_actions=3, epochs=10),
    CurriculumStage(num_states=5, max_actions=5, epochs=3),
]
```

### Model Comparison Interface

All models share a unified interface:

```python
from src.models.moore_transformer import MooreTransformer, TransformerConfig
from src.models.moore_vanilla_rnn import create_moore_vanilla_rnn  
from src.models.moore_lstm import create_moore_lstm

# Create models with same configuration
config = {"vocab_size": 21, "num_states": 5, "d_model": 256}

transformer = MooreTransformer(TransformerConfig(**config, num_heads=8))
vanilla_rnn = create_moore_vanilla_rnn(**config)
lstm = create_moore_lstm(**config)

# All models support the same interface
logits, loss = model(input_ids, targets=targets, unknown_mask=mask)
```

---

## 📈 Experimental Results

### Architecture Comparison (5-state, 5-action DFAs)

| Model | Parameters | Val Accuracy | Key Finding |
|-------|------------|--------------|-------------|
| Transformer (2L) | ~52K | **99.9%** | Requires curriculum learning |
| Large LSTM (256d, 1L) | ~531K | 98.9% | 10× parameters for similar accuracy |
| Small LSTM (64d, 2L) | ~68K | 84.2% | Plateaus at lower accuracy |
| Vanilla RNN | ~68K | ~Random | Failed to learn |

### Key Observations

1. **Phase Transitions**: Transformers exhibit sharp accuracy jumps (epochs 3-4), suggesting sudden discovery of correct attention patterns
2. **Curriculum Necessity**: Transformers fail on complex DFAs without pre-training on simple ones; LSTMs succeed with direct training but need more parameters
3. **Length Extrapolation**: LSTMs maintain accuracy on longer queries (200-600 tokens); Transformers degrade slightly

---

## 🧪 Testing & Validation

### Run Tests

```bash
# All tests
python -m pytest tests/ -v

# Model tests
python -m pytest tests/test_moore_models.py -v

# Trainer tests
python -m pytest tests/test_icl_trainer.py -v
```

### Test Coverage

- **Model Creation**: All architectures instantiate correctly
- **Forward Pass**: Correct output shapes and loss computation
- **Interface Consistency**: All models work interchangeably
- **Training Loop**: Gradient flow and optimization steps

---

## 📚 References

- **In-Context Learning**: Brown et al. (2020) - Language Models are Few-Shot Learners
- **Transformers**: Vaswani et al. (2017) - Attention is All You Need  
- **RoPE**: Su et al. (2021) - RoFormer: Enhanced Transformer with Rotary Position Embedding
- **Finite State Machines**: Hopcroft & Ullman (1979) - Introduction to Automata Theory
- **Complexity & FSMs**: Oprea (2020) - What Makes a Rule Complex?
- **Transformers & Formal Languages**: Akyürek et al. (2024) - In-Context Language Learning

---

## 🤝 Contributing

This project follows clean coding practices with:
- **Type hints** throughout the codebase
- **Comprehensive documentation** for all modules
- **Consistent interfaces** across components  
- **Full test coverage** for reliability

To extend the project:
1. Add new models in `src/models/` following the Moore interface
2. Create corresponding trainers in `src/training/`  
3. Add experiment scripts in `experiments/`
4. Include comprehensive tests in `tests/`

---

## 💡 Research Inspiration: Cognitive Complexity in Economics

This project was directly inspired by **Ryan Oprea's (2020)** work on measuring cognitive complexity in economic decision-making, published in the *American Economic Review*.

### The Economics Connection

Oprea's research investigates why human subjects in experimental economics perform poorly on certain decision tasks that appear computationally simple. His key insight: **the "complexity" of a decision problem is not about computational difficulty, but about the structure of the underlying state space.**

Oprea formalizes decision environments as **finite state machines** (specifically, Moore machines), where:
- **States** represent different environmental configurations
- **Actions** represent choices available to the decision-maker
- **Transitions** capture how the environment responds to actions

He finds that human performance correlates strongly with properties of the automaton structure—particularly the presence of **absorbing states** (states that trap the decision-maker) and **transition reversibility**.

### Why This Matters for AI & Economics

Our project extends this framework by asking: **Can neural networks learn the transition dynamics of finite automata from observation alone?**

This has direct applications to:
1. **Behavioral Economics**: Understanding how AI systems might model or predict human decision-making in complex environments
2. **Mechanism Design**: Designing economic mechanisms where AI agents must infer rules from examples
3. **Bounded Rationality**: Comparing neural network learning to human cognitive limitations
4. **Algorithmic Game Theory**: Training agents that can adapt to new strategic environments

### Key Findings Relevant to Economics

- **Transformers require curriculum learning** to master complex automata—paralleling findings that humans learn complex rules incrementally
- **Absorbing states** create learning challenges for neural networks, just as they do for human subjects
- **In-context learning** enables adaptation without retraining, similar to how humans generalize from examples

This connection between machine learning architectures and cognitive science provides a computational lens for understanding decision-making under complexity—a central concern in behavioral economics and finance.

### Reference

Oprea, R. (2020). What Makes a Rule Complex? *American Economic Review*, 110(12), 3913-3951.

---

*This project represents a systematic comparison of neural architectures on structured sequential learning, providing insights into the mechanisms underlying in-context learning across different model families.*
