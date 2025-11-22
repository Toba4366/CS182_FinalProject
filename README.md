# CS 182 Final Project: In-Context Learning of Moore Machines# CS 182 Final Project: In-Context Learning of Moore Machines



A comprehensive study of **In-Context Learning (ICL)** capabilities across different neural architectures when learning deterministic finite automata state transitions. This project compares how Transformers, LSTMs, and Vanilla RNNs perform on sequential pattern learning tasks in the context of Moore machines and general DFAs.A comprehensive study of **In-Context Learning (ICL)** capabilities across different neural architectures when learning deterministic finite automata state transitions. This project compares how Transformers, LSTMs, and Vanilla RNNs perform on sequential pattern learning tasks in the context of Moore machines and general DFAs.



**Team Members:** Trenton O'Bannon, Yuri Lee, Keshab Agarwal, Evan Davis



## 🎯 Project Overview## 🎯 Project Overview



### Motivation### Motivation



**In-Context Learning** is the remarkable ability of neural models to learn new patterns from a few examples provided in their input context, without updating their parameters. While this capability has been extensively studied in large language models (Transformers), little research exists on how different architectures compare on structured sequential learning tasks.**In-Context Learning** is the remarkable ability of neural models to learn new patterns from a few examples provided in their input context, without updating their parameters. While this capability has been extensively studied in large language models (Transformers), little research exists on how different architectures compare on structured sequential learning tasks.



**Deterministic Finite Automata (DFAs)** provide an ideal testbed because they represent:**Deterministic Finite Automata (DFAs)** provide an ideal testbed because they represent:

- **Deterministic state transitions**: Each state-action pair maps to exactly one next state- **Deterministic state transitions**: Each state-action pair maps to exactly one next state

- **Learnable patterns**: Complex enough for meaningful comparison, simple enough to analyze- **Learnable patterns**: Complex enough for meaningful comparison, simple enough to analyze

- **Mathematical structure**: Well-defined properties for validating model correctness- **Mathematical structure**: Well-defined properties for validating model correctness

- **Scalable complexity**: Configurable number of states and actions for progressive difficulty- **Scalable complexity**: Configurable number of states and actions for progressive difficulty



### Research Questions



1. **Architecture Comparison**: How do different neural architectures (Transformer, LSTM, Vanilla RNN) perform on ICL for finite state machines?### Research Questions

2. **Sequence Modeling**: Which architectural inductive biases are most effective for learning finite state machine patterns?

3. **ICL Mechanisms**: What makes certain architectures better at in-context learning for sequential decision tasks?1. **Architecture Comparison**: How do different neural architectures (Transformer, LSTM, Vanilla RNN) perform on ICL for finite state machines?

4. **DFA Properties**: How do different DFA characteristics (absorbing states, connectivity) affect learnability?2. **Sequence Modeling**: Which architectural inductive biases are most effective for learning finite state machine patterns?

3. **ICL Mechanisms**: What makes certain architectures better at in-context learning for sequential decision tasks?

## 🏗️ Project Architecture4. **DFA Properties**: How do different DFA characteristics (absorbing states, connectivity) affect learnability?



### Core Components## 🏗️ Project Architecture



```### Core Components

CS182_FinalProject/

├── src/                          # Source code```

│   ├── models/                   # Neural architecture implementationsCS182_FinalProject/

│   │   ├── moore_transformer.py  # Transformer with RoPE and causal attention├── src/                          # Source code

│   │   ├── moore_vanilla_rnn.py  # Vanilla RNN with tanh/relu activations│   ├── models/                   # Neural architecture implementations

│   │   └── moore_lstm.py         # LSTM with optional bidirectionality│   │   ├── moore_transformer.py  # Transformer with RoPE and causal attention

│   ├── training/                 # Model-specific trainers│   │   ├── moore_vanilla_rnn.py  # Vanilla RNN with tanh/relu activations

│   │   ├── icl_trainer.py        # Transformer ICL trainer│   │   └── moore_lstm.py         # LSTM with optional bidirectionality

│   │   ├── vanilla_rnn_trainer.py # Vanilla RNN ICL trainer  │   ├── training/                 # Model-specific trainers

│   │   └── lstm_trainer.py       # LSTM ICL trainer│   │   ├── icl_trainer.py        # Transformer ICL trainer

│   ├── datasets/                 # Data generation and loading│   │   ├── vanilla_rnn_trainer.py # Vanilla RNN ICL trainer  

│   │   └── moore_dataset.py      # DFA trajectory dataset for ICL│   │   └── lstm_trainer.py       # LSTM ICL trainer

│   ├── fsm/                      # Finite state machine utilities│   ├── datasets/                 # Data generation and loading

│   │   ├── generator.py          # DFA generation with deterministic transitions│   │   └── moore_dataset.py      # DFA trajectory dataset for ICL

│   │   └── trajectory_sampler.py # FSM trajectory sampling for ICL sequences│   ├── fsm/                      # Finite state machine utilities

│   └── utils/                    # Utility functions│   │   ├── generator.py          # DFA generation with deterministic transitions

├── experiments/                  # Training scripts│   │   └── trajectory_sampler.py # FSM trajectory sampling for ICL sequences

│   ├── run_icl_transformer.py    # Train Transformer model│   └── utils/                    # Utility functions

│   ├── run_icl_vanilla_rnn.py    # Train Vanilla RNN model  ├── experiments/                  # Training scripts

│   └── run_icl_lstm.py           # Train LSTM model│   ├── run_icl_transformer.py    # Train Transformer model

├── tests/                        # Comprehensive test suite│   ├── run_icl_vanilla_rnn.py    # Train Vanilla RNN model  

│   ├── test_moore_models.py      # Model functionality tests│   └── run_icl_lstm.py           # Train LSTM model

│   └── test_icl_trainer.py       # Trainer functionality tests├── tests/                        # Comprehensive test suite

└── data/                         # Generated datasets│   ├── test_moore_models.py      # Model functionality tests

    ├── icl_dataset.pt            # Pre-generated DFA trajectories│   └── test_icl_trainer.py       # Trainer functionality tests

    └── VOCAB.md                  # Vocabulary and tokenization docs└── data/                         # Generated datasets

```    ├── icl_dataset.pt            # Pre-generated DFA trajectories

    └── VOCAB.md                  # Vocabulary and tokenization docs

### Key Design Principles```



1. **Model-Agnostic ICL Interface**: All models implement the same `forward(input_ids, targets, unknown_mask)` signature for consistent evaluation### Key Design Principles

2. **Individual Trainers**: Each architecture has a dedicated trainer optimized for its specific requirements

3. **Comprehensive Testing**: Full test coverage ensures model correctness and interface consistency  1. **Model-Agnostic ICL Interface**: All models implement the same `forward(input_ids, targets, unknown_mask)` signature for consistent evaluation

4. **DFA Focus**: All components specialized for deterministic finite automata learning tasks2. **Individual Trainers**: Each architecture has a dedicated trainer optimized for its specific requirements

3. **Comprehensive Testing**: Full test coverage ensures model correctness and interface consistency  

## 🧠 Model Architectures4. **DFA Focus**: All components specialized for deterministic finite automata learning tasks



### 1. Moore Transformer (`moore_transformer.py`)## 🧠 Model Architectures

- **Decoder-only architecture** with causal attention

- **Rotary Position Embedding (RoPE)** for better sequence understanding  ### 1. Moore Transformer (`moore_transformer.py`)

- **Multi-head attention** with configurable heads and layers- **Decoder-only architecture** with causal attention

- **Specialized for ICL** with unknown state masking- **Rotary Position Embedding (RoPE)** for better sequence understanding  

- **Multi-head attention** with configurable heads and layers

### 2. Moore Vanilla RNN (`moore_vanilla_rnn.py`)- **Specialized for ICL** with unknown state masking

- **Elman network** with tanh/relu activations

- **Multi-layer support** with dropout between layers### 2. Moore Vanilla RNN (`moore_vanilla_rnn.py`)

- **Step-by-step processing** maintaining hidden states across timesteps

- **Minimal architecture** serving as baseline for comparison- **Elman network** with tanh/relu activations



### 3. Moore LSTM (`moore_lstm.py`)- **Multi-layer support** with dropout between layers## 🚀 Quick Start

- **Long Short-Term Memory** with forget/input/output gates

- **Bidirectional support** for enhanced context modeling- **Step-by-step processing** maintaining hidden states across timesteps

- **Proper initialization** with orthogonal hidden-to-hidden weights

- **Gradient-friendly** design for longer sequences- **Minimal architecture** serving as baseline for comparison1. **Install dependencies:**



## 🔧 DFA Generation   ```bash



### Deterministic Finite Automata Generator (`src/fsm/generator.py`)### 3. Moore LSTM (`moore_lstm.py`)   pip install -r requirements.txt



The updated generator creates **true deterministic finite automata** with the following properties:- **Long Short-Term Memory** with forget/input/output gates   ```



- **Deterministic Transitions**: Every state has exactly `action_count` outgoing transitions- **Bidirectional support** for enhanced context modeling

- **Configurable Complexity**: 

  - `num_states`: Number of states in the automaton (default: 5)- **Proper initialization** with orthogonal hidden-to-hidden weights2. **Test the setup:**

  - `min_actions`/`max_actions`: Range of actions per state (default: 3-8)

  - `action_count`: Computed as `num_states + 2` for proper vocabulary sizing- **Gradient-friendly** design for longer sequences   ```bash

- **Absorbing State Support**: Optional generation of DFAs with absorbing states

- **Connectivity**: Ensures all states are reachable from initial state   python scripts/test_imports.py    # Test basic imports



### Key Improvements## 📊 In-Context Learning Setup   python scripts/run_quick_training.py  # Run a quick training



- **Instance-Based Configuration**: Each `FSMGeneratorConfig` computes its own `action_count` via `@property`   ```

- **Data Integrity**: No shared state between different generator configurations  

- **Mathematical Correctness**: Validates DFA properties (exactly `action_count` transitions per state)### Sequence Format

- **Extensibility**: Support for both standard and absorbing state DFAs

3. **View training curves:**

## 📊 In-Context Learning Setup

Each training sequence follows the ICL paradigm:   ```bash

### Sequence Format

```   jupyter notebook notebooks/training_analysis.ipynb

Each training sequence follows the ICL paradigm:

```Demo Examples: S₁, A₁ → S₂, A₂ → S₃, ..., <eos>   # Or run in VS Code with Jupyter extension

Demo Examples: S₁, A₁ → S₂, A₂ → S₃, ..., <eos>

Query Segment: S₇, A₇ → <unk>, A₈ → <unk>, A₉ → <unk>Query Segment: S₇, A₇ → <unk>, A₈ → <unk>, A₉ → <unk>   ```

```

```

Where:

- **Demo Examples**: Known state transitions the model learns from4. **Run experiments:**

- **Query Segment**: Unknown states (`<unk>`) the model must predict

- **Loss Masking**: Only unknown positions contribute to lossWhere:   ```bash



### Vocabulary Structure- **Demo Examples**: Known state transitions the model learns from   # 2-layer transformer (default)



- **State Tokens**: `0 ... num_states-1`- **Query Segment**: Unknown states (`<unk>`) the model must predict   python scripts/run_quick_training.py

- **Action Tokens**: `num_states ... num_states + action_count - 1`  

- **Special Tokens**:- **Loss Masking**: Only unknown positions contribute to loss   

  - `<eos>`: Separates demo from query segments

  - `<query>`: Marks start of query portion   # Or use the full experiment runner:

  - `<pad>`: Padding for batch processing

### Vocabulary Structure   python -m experiments.run_experiment --config configs/base_config.yaml

### Dataset Generation

   

The `MooreICLDataset` creates sequences by:

1. **Generating diverse DFAs** with varying complexity using the deterministic generator- **State Tokens**: `0 ... num_states-1`   # 3-layer comparison

2. **Sampling demonstration trajectories** showing valid state transitions

3. **Creating query segments** with unknown states to predict- **Action Tokens**: `num_states ... num_states + max_actions - 1`     python -m experiments.run_experiment --config configs/3layer_config.yaml

4. **Applying loss masking** to focus learning on unknowns

- **Special Tokens**:   

## 🚀 Usage

  - `<eos>`: Separates demo from query segments   # Frozen layer experiment (only final layer trains)

### Training Models

  - `<query>`: Marks start of query portion   python -m experiments.run_experiment --config configs/frozen_layers_config.yaml

```bash

# Train Transformer (default model)  - `<pad>`: Padding for batch processing   ```

python experiments/run_icl_transformer.py --epochs 10 --batch-size 8 --num-layers 4



# Train Vanilla RNN with verbose logging

python experiments/run_icl_vanilla_rnn.py --epochs 10 --d-model 256 --activation tanh### Dataset Generation## 🧪 Experimental Configurations



# Train LSTM with bidirectional processing

python experiments/run_icl_lstm.py --epochs 10 --d-model 256 --bidirectional

The `MooreICLDataset` creates sequences by:### Base Configuration (`configs/base_config.yaml`)

# Disable verbose logging (default is enabled)

python experiments/run_icl_vanilla_rnn.py --no-verbose1. **Generating diverse Moore machines** with varying complexity- 2-layer transformer with 4 attention heads

```

2. **Sampling demonstration trajectories** showing state transitions- 128 model dimension, optimized for efficiency

### Verbose Training Features

3. **Creating query segments** with unknown states to predict- Standard training with all parameters trainable

All RNN and LSTM trainers support detailed logging showing:

- **Input sequence shapes and content**4. **Applying loss masking** to focus learning on unknowns

- **Target states and loss masks**  

- **Unknown position counts**### 3-Layer Configuration (`configs/3layer_config.yaml`) 

- **Loss values and training progress**

## 🚀 Usage- 3-layer transformer for comparison

Example verbose output:

```- Same hyperparameters for fair comparison

🔍 [Epoch 1] Detailed Batch Info:

   Input IDs shape: torch.Size([8, 64])### Training Models

   Input IDs (first sample): tensor([1, 2, 3, 4, 5, 6, 0, 7, 8, ...])

   Target IDs (first sample): tensor([2, 3, 4, 5, 6, 0, 7, 8, 9, ...])### Frozen Layer Configuration (`configs/frozen_layers_config.yaml`)

   Loss mask (first sample): tensor([False, False, ..., True, True])

   Loss: 1.2345```bash- **Tests core hypothesis**: Can only the final linear layer solve ICL?

   Unknown positions: 16 out of 64

```# Train Transformer (default model)- Freezes all transformer layers and embeddings



### Model Comparisonpython experiments/run_icl_transformer.py --epochs 10 --batch-size 8 --num-layers 4- Only the `lm_head` (final linear layer) remains trainable



```python

from src.models.moore_transformer import MooreTransformer, TransformerConfig

from src.models.moore_vanilla_rnn import create_moore_vanilla_rnn  # Train Vanilla RNN  ## 🔬 Key Features

from src.models.moore_lstm import create_moore_lstm

python experiments/run_icl_vanilla_rnn.py --epochs 10 --d-model 256 --activation tanh

# Create models with same configuration

config = {"vocab_size": 20, "num_states": 5, "d_model": 256}### Moore Machine Implementation



transformer = MooreTransformer(TransformerConfig(**config, num_heads=8))# Train LSTM- Exactly 5 states (constraint from project scope)

vanilla_rnn = create_moore_vanilla_rnn(**config)

lstm = create_moore_lstm(**config)python experiments/run_icl_lstm.py --epochs 10 --d-model 256 --bidirectional- Variable 5-8 actions per machine



# All models support the same interface- 4-8 state transitions including self-loops

logits, loss = model(input_ids, targets=targets, unknown_mask=mask)

```# Disable verbose logging- Automatic validation of constraint compliance



### Running Testspython experiments/run_icl_vanilla_rnn.py --no-verbose



```bash```### Multi-Architecture Implementation

# Run all tests

python -m pytest tests/ -v**Transformer (Baseline)**



# Test specific components### Model Comparison- Decoder-only architecture with causal masking

python -m pytest tests/test_moore_models.py::TestModelComparison -v

python -m pytest tests/test_icl_trainer.py -v- Multi-head attention with positional encoding

```

```python- Parameters: ~729k (d=128, 2L)

## 🔬 Research Insights

from src.models.moore_transformer import MooreTransformer, TransformerConfig- File: `traditional.py`

### Expected Findings

from src.models.moore_vanilla_rnn import create_moore_vanilla_rnn  

1. **Transformer Advantages**: Superior ICL due to attention mechanism's ability to directly relate query positions to relevant demo examples

2. **LSTM vs Vanilla RNN**: LSTM's gating mechanisms should provide better long-range dependency modeling for complex DFAsfrom src.models.moore_lstm import create_moore_lstm**Vanilla RNN** 

3. **Sequence Length Effects**: Longer context windows should benefit Transformers more than recurrent models

4. **Architecture Scaling**: Different optimal model sizes for each architecture- Basic Elman network with tanh activation

5. **DFA Complexity**: Absorbing states and connectivity patterns may create different learning challenges

# Create models with same configuration- Simple recurrent connections, minimal parameters

### Evaluation Metrics

config = {"vocab_size": 20, "num_states": 5, "d_model": 256}- Parameters: ~75k (d=128, 2L)

- **Accuracy**: Percentage of correctly predicted unknown states

- **Loss Convergence**: Training dynamics across architectures- File: `vanilla_rnn.py`

- **Parameter Efficiency**: Performance per parameter count

- **Sequence Length Scaling**: Performance vs context lengthtransformer = MooreTransformer(TransformerConfig(**config, num_heads=8))

- **DFA Property Learning**: Ability to maintain deterministic transition patterns

vanilla_rnn = create_moore_vanilla_rnn(**config)**LSTM**

## 🛠️ Development Features

lstm = create_moore_lstm(**config)- Long Short-Term Memory with gating mechanisms

### Comprehensive Testing

- **25 test cases** covering model creation, training, and interfaces- Enhanced memory and gradient flow

- **Mock datasets** for fast testing without full data generation

- **Interface consistency** tests ensuring all models work interchangeably# All models support the same interface- Parameters: ~273k (d=128, 2L)



### DFA Validationlogits, loss = model(input_ids, targets=targets, unknown_mask=mask)- File: `lstm.py`

- **Deterministic property checking**: Validates exactly `action_count` transitions per state

- **Connectivity verification**: Ensures all states are reachable```

- **Configuration integrity**: Instance-based properties prevent shared state bugs

**Training Features**

### Modular Design

- **Clean separation** between models, trainers, and experiments### Running Tests- Configurable freezing for ablation studies

- **Reusable components** for dataset generation and evaluation

- **Extensible architecture** for adding new model types- Parameter counting and frozen parameter tracking



## 📈 Future Extensions```bash- Unified interface across all architectures



1. **State Space Models**: Add Mamba/S4 architectures for comparison# Run all tests

2. **Curriculum Learning**: Progressive difficulty in DFA complexity

3. **Transfer Learning**: Pre-training on simpler DFAs, fine-tuning on complex onespython -m pytest tests/ -v### Training Framework

4. **Attention Analysis**: Visualizing what Transformers attend to during ICL

5. **Scaling Laws**: Model size vs ICL performance relationships- AdamW optimizer with warmup and cosine annealing

6. **DFA Variations**: Non-deterministic automata, epsilon transitions

# Test specific components- Gradient clipping and automatic checkpointing

## 🤝 Contributing

python -m pytest tests/test_moore_models.py::TestModelComparison -v- Optional Weights & Biases integration

This project follows clean coding practices with:

- **Type hints** throughout the codebasepython -m pytest tests/test_icl_trainer.py -v- Comprehensive evaluation metrics

- **Comprehensive documentation** for all modules

- **Consistent interfaces** across components  ```

- **Full test coverage** for reliability

### Visualization Tools

To extend the project:

1. Add new models in `src/models/` following the Moore interface## 🔬 Research Insights- FSM diagram generation with NetworkX

2. Create corresponding trainers in `src/training/`  

3. Add experiment scripts in `experiments/`- Training curve plotting

4. Include comprehensive tests in `tests/`

### Expected Findings- Attention pattern visualization

## 📚 References

- Performance analysis utilities

- **In-Context Learning**: Brown et al. (2020) - Language Models are Few-Shot Learners

- **Transformers**: Vaswani et al. (2017) - Attention is All You Need  1. **Transformer Advantages**: Superior ICL due to attention mechanism's ability to directly relate query positions to relevant demo examples

- **RoPE**: Su et al. (2021) - RoFormer: Enhanced Transformer with Rotary Position Embedding

- **Moore Machines**: Moore (1956) - Gedanken-experiments on Sequential Machines2. **LSTM vs Vanilla RNN**: LSTM's gating mechanisms should provide better long-range dependency modeling for complex FSMs  ## 📊 Planned Experiments

- **Finite State Machines**: Hopcroft & Ullman (1979) - Introduction to Automata Theory

3. **Sequence Length Effects**: Longer context windows should benefit Transformers more than recurrent models

---

4. **Architecture Scaling**: Different optimal model sizes for each architecture1. **Baseline Performance**: 2-layer vs 3-layer transformers

This project represents a systematic comparison of neural architectures on structured sequential learning, providing insights into the mechanisms underlying in-context learning across different model families. The updated DFA generator ensures mathematical correctness and proper deterministic automata for reliable experimental results.
2. **Frozen Layer Analysis**: Test if only final layer can solve ICL

### Evaluation Metrics3. **Scaling Studies**: Model size vs performance trade-offs

4. **Complexity Analysis**: FSM complexity vs learning difficulty

- **Accuracy**: Percentage of correctly predicted unknown states5. **Attention Visualization**: What patterns do transformers learn?

- **Loss Convergence**: Training dynamics across architectures

- **Parameter Efficiency**: Performance per parameter count## 🛠️ Development Notes

- **Sequence Length Scaling**: Performance vs context length

This scaffolding was generated to provide:

## 🛠️ Development Features- ✅ Complete project structure with working imports

- ✅ Constrained FSM generation matching project requirements

### Comprehensive Testing- ✅ Small transformer models (2-3 layers) for efficient experimentation

- **25 test cases** covering model creation, training, and interfaces- ✅ Frozen parameter experiments for mechanistic analysis

- **Mock datasets** for fast testing without full data generation- ✅ AdamW-only optimization (reduced scope)

- **Interface consistency** tests ensuring all models work interchangeably- ✅ Comprehensive configuration system

- ✅ Ready-to-run examples and training visualization

### Verbose Training- ✅ Jupyter notebook for training analysis

All trainers support detailed logging showing:

- Input sequence shapes and content**Latest Update**: Updated FSM solver to support both tuple and class interfaces, aligned sequence/context lengths with project plan (64 sequences, 256 context), and implemented universal compatibility for multi-architecture development.

- Target states and loss masks  

- Unknown position counts## 📊 Complete 10,000 Sample Dataset Generated

- Loss values and training progress

**✨ NEW: Multi-Format Dataset System**

### Modular Design

- **Clean separation** between models, trainers, and experimentsWe've generated a complete 10,000 sample dataset as specified in `plan.md`, available in 4 different formats to suit different workflows:

- **Reusable components** for dataset generation and evaluation

- **Extensible architecture** for adding new model types### 📁 Available Dataset Formats



## 📈 Future Extensions| Format | Size | Best For | Location |

|--------|------|----------|----------|

1. **State Space Models**: Add Mamba/S4 architectures for comparison| **PKL** | ~19MB | 🏃 Fastest Python training | `data/full_dataset_pkl/` |

2. **Curriculum Learning**: Progressive difficulty in Moore machine complexity| **JSON** | ~144MB | 👁️ Human inspection & debugging | `data/full_dataset_json/` |

3. **Transfer Learning**: Pre-training on simpler FSMs, fine-tuning on complex ones| **Parquet** | ~6.8MB | 🏢 Production & data analysis | `data/full_dataset_parquet/` |

4. **Attention Analysis**: Visualizing what Transformers attend to during ICL| **HDF5** | ~16MB | 🔬 Scientific computing | `data/full_dataset_hdf5/` |

5. **Scaling Laws**: Model size vs ICL performance relationships

### 📈 Dataset Statistics

## 🤝 Contributing- **Training**: 6,000 samples  

- **Validation**: 2,000 samples

This project follows clean coding practices with:- **Test**: 2,000 samples

- **Type hints** throughout the codebase- **Total**: 10,000 samples (matching plan.md specifications)

- **Comprehensive documentation** for all modules- **Truncation Distribution**: ~25% start_state, ~50% action, ~25% non_start_state

- **Consistent interfaces** across components  

- **Full test coverage** for reliability### 🚀 Quick Dataset Usage



To extend the project:**Choose your preferred format:**

1. Add new models in `src/models/` following the Moore interface

2. Create corresponding trainers in `src/training/`  ```python

3. Add experiment scripts in `experiments/`# Option A: Fastest training (PKL)

4. Include comprehensive tests in `tests/`from tests.test_data_integrity import FSMDataset_PKL

dataset = FSMDataset_PKL('./data/full_dataset_pkl', 'train')

## 📚 References

# Option B: Most readable (JSON) 

- **In-Context Learning**: Brown et al. (2020) - Language Models are Few-Shot Learnersfrom tests.test_data_integrity import FSMDataset_JSON

- **Transformers**: Vaswani et al. (2017) - Attention is All You Need  dataset = FSMDataset_JSON('./data/full_dataset_json', 'train')

- **RoPE**: Su et al. (2021) - RoFormer: Enhanced Transformer with Rotary Position Embedding

- **Moore Machines**: Moore (1956) - Gedanken-experiments on Sequential Machines# Option C: Production ready (Parquet)

from tests.test_data_integrity import FSMDataset_Parquet  

---dataset = FSMDataset_Parquet('./data/full_dataset_parquet', 'train')



This project represents a systematic comparison of neural architectures on structured sequential learning, providing insights into the mechanisms underlying in-context learning across different model families.# Option D: Scientific computing (HDF5)
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

### Multi-Architecture Testing (`tests/test_multi_architecture.py`) 
**🔍 NEW: Comprehensive cross-architecture validation**

Tests all model architectures (Transformer, Vanilla RNN, LSTM) for:

1. **Token Compatibility**: 36-token vocabulary handling across all architectures
2. **Training Functionality**: Forward/backward passes, loss computation, optimizer steps
3. **Parameter Analysis**: Count comparison and memory efficiency analysis
4. **Dataset Integration**: All formats (PKL/JSON/Parquet/HDF5) work with all models  
5. **Gradient Health**: Gradient flow validation and numerical stability checks

```bash
# Run multi-architecture test suite
python tests/test_multi_architecture.py
```

**Results**: ✅ 5/5 tests passing across 3 architectures
- Transformer: 729k params, strong attention baseline
- Vanilla RNN: 75k params (0.1x), simple recurrent baseline  
- LSTM: 273k params (0.4x), enhanced memory & gating

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
