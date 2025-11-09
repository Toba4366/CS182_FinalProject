# CS 182 Final Project: In-Context Learning of Moore Machines

**🚧 GITHUB COPILOT SCAFFOLDING - TO BE EDITED AND CHANGED LATER 🚧**

This repository contains the scaffolding for our CS 182 final project on in-context learning of finite state machines using transformer models. This initial framework was generated with GitHub Copilot to provide a solid foundation for team collaboration.

## 🎯 Project Overview

We study how transformer models can learn to simulate Moore machines through in-context learning, focusing on:

- **Constrained FSM Parameters**: 5 states, 5-8 actions, 4-8 transitions with self-loops
- **Small Transformer Models**: Optimized for 2-3 layer experiments
- **AdamW Optimizer**: Single optimizer focus to reduce experimental scope
- **Frozen Layer Experiments**: Test whether only the final linear layer can solve ICL

## 🏗️ Repository Structure

```
CS182_FinalProject/
├── src/
│   ├── fsm/              # Moore machine implementation
│   ├── training/         # PyTorch models & training loops
│   ├── utils/           # Visualization & analysis tools
│   └── experiments/     # Experiment runners
├── configs/             # YAML configuration files
├── scripts/            # Training automation & testing
├── notebooks/          # Jupyter notebooks for exploration
├── papers/             # Research papers and references
└── requirements.txt    # Python dependencies
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the setup:**
   ```bash
   python scripts/test_imports.py    # Test basic imports
   python scripts/run_quick_training.py  # Run a quick training
   ```

3. **View training curves:**
   ```bash
   jupyter notebook notebooks/training_analysis.ipynb
   # Or run in VS Code with Jupyter extension
   ```

4. **Run experiments:**
   ```bash
   # 2-layer transformer (default)
   python scripts/run_quick_training.py
   
   # Or use the full experiment runner:
   python -m experiments.run_experiment --config configs/base_config.yaml
   
   # 3-layer comparison
   python -m experiments.run_experiment --config configs/3layer_config.yaml
   
   # Frozen layer experiment (only final layer trains)
   python -m experiments.run_experiment --config configs/frozen_layers_config.yaml
   ```

## 🧪 Experimental Configurations

### Base Configuration (`configs/base_config.yaml`)
- 2-layer transformer with 4 attention heads
- 128 model dimension, optimized for efficiency
- Standard training with all parameters trainable

### 3-Layer Configuration (`configs/3layer_config.yaml`) 
- 3-layer transformer for comparison
- Same hyperparameters for fair comparison

### Frozen Layer Configuration (`configs/frozen_layers_config.yaml`)
- **Tests core hypothesis**: Can only the final linear layer solve ICL?
- Freezes all transformer layers and embeddings
- Only the `lm_head` (final linear layer) remains trainable

## 🔬 Key Features

### Moore Machine Implementation
- Exactly 5 states (constraint from project scope)
- Variable 5-8 actions per machine
- 4-8 state transitions including self-loops
- Automatic validation of constraint compliance

### Transformer Architecture
- Decoder-only architecture with causal masking
- Multi-head attention with positional encoding
- Configurable freezing for ablation studies
- Parameter counting and frozen parameter tracking

### Training Framework
- AdamW optimizer with warmup and cosine annealing
- Gradient clipping and automatic checkpointing
- Optional Weights & Biases integration
- Comprehensive evaluation metrics

### Visualization Tools
- FSM diagram generation with NetworkX
- Training curve plotting
- Attention pattern visualization
- Performance analysis utilities

## 📊 Planned Experiments

1. **Baseline Performance**: 2-layer vs 3-layer transformers
2. **Frozen Layer Analysis**: Test if only final layer can solve ICL
3. **Scaling Studies**: Model size vs performance trade-offs
4. **Complexity Analysis**: FSM complexity vs learning difficulty
5. **Attention Visualization**: What patterns do transformers learn?

## 🛠️ Development Notes

This scaffolding was generated to provide:
- ✅ Complete project structure with working imports
- ✅ Constrained FSM generation matching project requirements
- ✅ Small transformer models (2-3 layers) for efficient experimentation
- ✅ Frozen parameter experiments for mechanistic analysis
- ✅ AdamW-only optimization (reduced scope)
- ✅ Comprehensive configuration system
- ✅ Ready-to-run examples and training visualization
- ✅ Jupyter notebook for training analysis

**Latest Update**: Fixed all import path issues, implemented proper loss computation for in-context learning, and validated training pipeline with working visualization.

## � Team-Generated Experimental Extensions

**NEW: Beyond the Scaffold** - The team has developed additional FSM implementations that extend our research capabilities:

### `fsm_generator.py` - General DFA Generation
**Purpose**: Generate deterministic finite automata with different constraints than the Moore machines in our scaffold.

**Key Differences from Scaffold**:
- **Variable state counts**: 4-5 states (vs. exactly 5 in Moore machines)
- **Broader action range**: 2-8 actions (vs. 5-8 in scaffold)
- **Complete transition function**: Every (state, action) pair defined (vs. partial 4-8 transitions)
- **No explicit outputs**: Pure state transitions (vs. Moore machine outputs)
- **Absorption control**: Option to prevent/allow absorbing states

### `fsm_solver.py` - Partial Observation Framework
**Purpose**: Create sequences that start from different execution points, simulating partial observation scenarios.

**Core Innovation**: **Truncation Strategy for Partial Observation**
- **25%** start with start state (remove 0-3 complete transitions)
- **50%** start with action (mid-execution entry points)
- **25%** start with non-start state (arbitrary state entry)

**Training Format Flexibility**:
```python
format_sequence_for_training(path, "state_action_state")  # Ss0 A1 Ss1 Ss1 A2 Ss3
format_sequence_for_training(path, "action_only")         # A1 A2 A3
format_sequence_for_training(path, "state_only")          # Ss0 Ss1 Ss3
```

### 🎯 How This Advances Our Research Goals

**1. Comparative FSM Analysis**
- **Scaffold**: Moore machines with state-based outputs
- **New**: General DFAs with pure state transitions
- **Research Impact**: Test whether output-based vs. pure-transition tasks affect ICL difficulty

**2. Partial Observation ICL**
- **Scaffold**: Complete action-output sequences from start
- **New**: Truncated sequences starting mid-execution
- **Research Impact**: Study how transformers handle incomplete context in ICL

**3. Mechanistic Understanding**
- **Scaffold**: Fixed action-output alternating format
- **New**: Multiple sequence formats (action-only, state-only, mixed)
- **Research Impact**: Isolate which sequence components drive ICL performance

### 🚀 Future Research Directions

**1. Hybrid Experiments**
```python
# Compare Moore vs. DFA on same transformer
moore_results = train_on_moore_machines()
dfa_results = train_on_general_dfas()
analyze_complexity_differences()
```

**2. Partial Context Studies**
```python
# Test ICL robustness to incomplete information
full_context_performance = train_full_sequences()
partial_context_performance = train_truncated_sequences()
measure_context_dependency()
```

**3. Format Ablation Studies**
```python
# Which sequence elements matter most?
action_only_performance = train_format("action_only")
state_only_performance = train_format("state_only")  
mixed_performance = train_format("state_action_state")
```

**4. Absorption vs. Non-Absorption**
- Test whether absorbing states create "easier" ICL tasks
- Study transformer attention patterns on different FSM structures

This team-generated extension provides a **complementary experimental framework** to our Moore machine scaffold, enabling deeper investigation into the mechanistic basis of in-context learning with structured sequences.

## �👥 Team Collaboration

**This is initial scaffolding generated by GitHub Copilot.** The framework is designed to be:
- **Modular**: Easy to modify individual components
- **Configurable**: YAML-based experiment configuration
- **Extensible**: Clean interfaces for adding new features
- **Well-documented**: Comprehensive docstrings and comments

Feel free to modify, extend, or completely rewrite any part of this codebase as needed for our research goals!

## 📝 Next Steps

1. ✅ **Setup Complete**: All imports working, training pipeline validated
2. **Run baseline experiments** to confirm full functionality
3. **Test frozen layer hypothesis** using `configs/frozen_layers_config.yaml`
4. **Analyze training curves** in the provided Jupyter notebook
5. **Extend analysis tools** based on experimental needs
6. **Add team-specific modifications** and improvements
7. **Scale up experiments** as computational resources allow

## 🔧 **Troubleshooting**

- **Import errors**: Run `python scripts/test_imports.py` to validate setup
- **Training issues**: Try `python scripts/run_quick_training.py` for a minimal test
- **Visualization**: Use `notebooks/training_analysis.ipynb` for plotting
- **Configuration**: Check YAML configs in `configs/` directory

---

*Generated with GitHub Copilot as starting scaffold - ready for team development!*