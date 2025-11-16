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
├── data/                    # 📁 Generated datasets (multiple formats available!)
│   ├── full_dataset_pkl/    # 🏃 Fastest - use for training  
│   ├── full_dataset_json/   # 👁️ Most readable - use for debugging
│   ├── full_dataset_parquet/# 🏢 Most compressed - use for production
│   └── full_dataset_hdf5/   # 🔬 Scientific - use for massive scale
├── src/
│   ├── fsm/              # Moore machine implementation
│   ├── training/         # PyTorch models & training loops
│   └── utils/           # Visualization & analysis tools
├── utils/               # 🛠️ Dataset generation & conversion utilities
│   ├── generate_dataset.py  # Generate datasets in multiple formats
│   └── convert_dataset.py   # Convert between different formats
├── tests/               # 🧪 Testing & validation
│   ├── test_data_integrity.py  # Verify data quality across formats
│   └── test_training_pipeline.py  # COMPLETE training pipeline validation
├── experiments/         # Experiment runners
├── configs/             # YAML configuration files
├── scripts/            # Training automation & testing
├── notebooks/          # Jupyter notebooks for exploration
├── papers/             # Research papers and references
└── requirements.txt    # Python dependencies (updated for all formats)
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

**Latest Update**: Updated FSM solver to support both tuple and class interfaces, aligned sequence/context lengths with project plan (64 sequences, 256 context), and implemented universal compatibility for multi-architecture development.

## 📊 Complete 10,000 Sample Dataset Generated

**✨ NEW: Multi-Format Dataset System**

We've generated a complete 10,000 sample dataset as specified in `plan.md`, available in 4 different formats to suit different workflows:

### 📁 Available Dataset Formats

| Format | Size | Best For | Location |
|--------|------|----------|----------|
| **PKL** | ~19MB | 🏃 Fastest Python training | `data/full_dataset_pkl/` |
| **JSON** | ~144MB | 👁️ Human inspection & debugging | `data/full_dataset_json/` |
| **Parquet** | ~6.8MB | 🏢 Production & data analysis | `data/full_dataset_parquet/` |
| **HDF5** | ~16MB | 🔬 Scientific computing | `data/full_dataset_hdf5/` |

### 📈 Dataset Statistics
- **Training**: 6,000 samples  
- **Validation**: 2,000 samples
- **Test**: 2,000 samples
- **Total**: 10,000 samples (matching plan.md specifications)
- **Truncation Distribution**: ~25% start_state, ~50% action, ~25% non_start_state

### 🚀 Quick Dataset Usage

**Choose your preferred format:**

```python
# Option A: Fastest training (PKL)
from tests.test_data_integrity import FSMDataset_PKL
dataset = FSMDataset_PKL('./data/full_dataset_pkl', 'train')

# Option B: Most readable (JSON) 
from tests.test_data_integrity import FSMDataset_JSON
dataset = FSMDataset_JSON('./data/full_dataset_json', 'train')

# Option C: Production ready (Parquet)
from tests.test_data_integrity import FSMDataset_Parquet  
dataset = FSMDataset_Parquet('./data/full_dataset_parquet', 'train')

# Option D: Scientific computing (HDF5)
from tests.test_data_integrity import FSMDataset_HDF5
dataset = FSMDataset_HDF5('./data/full_dataset_hdf5', 'train')
```

**Validate everything works:**
```bash
python tests/test_data_integrity.py  # Tests all formats + performance benchmarks
python tests/test_training_pipeline.py  # COMPLETE training pipeline validation
```

**Generate custom datasets:**
```bash
python utils/generate_dataset.py --format pkl --output-dir ./my_dataset
python utils/convert_dataset.py --input-dir ./data/full_dataset_pkl --output-dir ./converted
```

## 🧪 Testing & Validation

### Data Integrity Testing (`tests/test_data_integrity.py`)
- Validates identical data across all 4 formats
- Performance benchmarks for format selection
- Complete Dataset class examples for PyTorch integration

### Training Pipeline Validation (`tests/test_training_pipeline.py`)
**🚀 CRITICAL: Run this before any serious training!**

Comprehensive validation of the complete training pipeline:

1. **Token Compatibility**: Ensures 36-token vocabulary aligns between dataset and transformer
2. **Single Batch Training**: Validates forward/backward passes and loss computation
3. **Multi-Format Integration**: Tests all dataset formats work with DataLoader
4. **Frozen Layer Setup**: Validates ICL experiment infrastructure (freeze all except final head)
5. **Learning Rate Schedule**: Tests warmup + cosine decay scheduler functionality

```bash
# Run complete validation suite
python tests/test_training_pipeline.py

# Expected output: 5/5 tests passed ✅
# Token Compatibility ✅ 
# Single Batch Training ✅ (loss decreases ~12%)
# Dataset Format Integration ✅ (all 4 formats work)
# Frozen Layer Setup ✅ (ICL experiment ready)
# Learning Rate Schedule ✅ (warmup → cosine decay)
```

**If all tests pass**: Ready for training experiments!  
**If tests fail**: Fix infrastructure issues before proceeding.

## 🔬 Team-Generated Experimental Extensions

**UPDATED: Beyond the Scaffold** - The team has developed additional FSM implementations and aligned with project plan specifications:

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

**4. Learning Rate Schedule Experiments**
```python
# Compare optimization strategies for FSM learning
from traditional import create_warmup_cosine_scheduler

# Baseline: Warmup + Cosine Decay (current)
baseline_scheduler = create_warmup_cosine_scheduler(optimizer, 1000, 10000)

# Experiment 1: Different warmup lengths
short_warmup = create_warmup_cosine_scheduler(optimizer, 500, 10000)   # 5% warmup
long_warmup = create_warmup_cosine_scheduler(optimizer, 2000, 10000)   # 20% warmup

# Experiment 2: Constant learning rate comparison  
constant_lr_optimizer = AdamW(model.parameters(), lr=3e-4)  # No scheduling

# Experiment 3: Step decay schedule
step_scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)

# Research questions:
# - Does warmup matter for small FSM transformers?
# - Do different schedules affect which FSM patterns are learned?
# - How does optimization strategy impact frozen layer ICL experiments?
# - Which schedule works best for partial observability scenarios?
```

**5. Architecture Comparisons**
```python
# Baseline transformers vs. other architectures
transformer_results = train_transformer_baseline()
rnn_results = train_rnn_baseline()
lstm_results = train_lstm_baseline()
# Future: S4/Mamba for sequence modeling comparison
```

**6. Absorption vs. Non-Absorption**
- Test whether absorbing states create "easier" ICL tasks
- Study transformer attention patterns on different FSM structures

**7. Mechanistic Interpretability**
```python
# Analyze what each model component learns
attention_patterns = visualize_attention_heads(model, fsm_sequences)
layer_representations = analyze_hidden_states_by_layer(model)
frozen_vs_full_comparison = compare_learned_representations()
```

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
2. ✅ **Pipeline Tested**: Complete validation suite passes (5/5 tests)
3. **Run baseline experiments** to confirm full functionality
4. **Test frozen layer hypothesis** using `configs/frozen_layers_config.yaml`
5. **Compare learning rate schedules** for FSM optimization
6. **Analyze training curves** in the provided Jupyter notebook
7. **Extend analysis tools** based on experimental needs
8. **Add team-specific modifications** and improvements
9. **Scale up experiments** as computational resources allow

## 🔧 **Troubleshooting**

- **Import errors**: Run `python scripts/test_imports.py` to validate setup
- **Training issues**: Run `python tests/test_training_pipeline.py` for comprehensive validation
- **Data format issues**: Run `python tests/test_data_integrity.py` to verify dataset consistency
- **Visualization**: Use `notebooks/training_analysis.ipynb` for plotting
- **Configuration**: Check YAML configs in `configs/` directory

**Before any training**: Ensure `tests/test_training_pipeline.py` passes all 5 tests!

---

*Generated with GitHub Copilot as starting scaffold - ready for team development!*