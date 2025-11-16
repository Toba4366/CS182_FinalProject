#!/usr/bin/env python3
"""
FSM Dataset Generation Script

This script generates training datasets for finite state machine (FSM) in-context learning
experiments. It creates sequences of (state, action) pairs with different truncation strategies
to simulate partial observability scenarios.

Key Features:
- Supports multiple output formats: PKL, JSON, Parquet, HDF5
- Implements three truncation modes: start_state (25%), action (50%), non_start_state (25%)
- Generates tokenized sequences suitable for transformer training
- Creates comprehensive metadata and summaries for analysis

Usage:
    python utils/generate_dataset.py --format json --train-samples 6000 --val-samples 2000 --test-samples 2000

The script generates 10,000 total samples as specified in plan.md:
- 6,000 training samples
- 2,000 validation samples  
- 2,000 test samples

Each sample contains:
- Tokenized sequence of (state, action) pairs
- FSM structure and execution details
- Truncation metadata for partial observability analysis
- Original and truncated execution paths
"""

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import numpy as np

from fsm_generator import generate_random_fsms
from fsm_solver import generate_truncated_sequence

# Import save functions from convert_dataset
import sys
sys.path.append('.')
from convert_dataset import (
    save_dataset_pkl, save_dataset_json, 
    save_dataset_parquet, save_dataset_hdf5
)


def set_seed(seed: int):
    """
    Set random seed for reproducibility across all random number generators.
    
    Args:
        seed: Random seed value for deterministic dataset generation
    """
    random.seed(seed)
    np.random.seed(seed)


def create_vocabulary_mapping() -> Dict[str, int]:
    """
    Create vocabulary mapping for tokenizing FSM sequences.
    
    The vocabulary includes:
    - Special tokens: <PAD>, <BOS>, <EOS>, <SEP>
    - State tokens: s0, s1, ..., s15 (16 states)
    - Action tokens: a0, a1, ..., a15 (16 actions)
    
    Total vocabulary size: 4 + 16 + 16 = 36 tokens
    
    Returns:
        Dict mapping token strings to integer IDs
        
    Note:
        This creates the tokenization scheme for (state, action) pairs as 
        described in plan.md. Each FSM execution step becomes two tokens.
    """
    vocab = {}
    token_id = 0
    
    # Special tokens
    vocab['<PAD>'] = token_id; token_id += 1
    vocab['<BOS>'] = token_id; token_id += 1  # Beginning of sequence
    vocab['<EOS>'] = token_id; token_id += 1  # End of sequence
    vocab['<SEP>'] = token_id; token_id += 1  # Separator between demo and test
    
    # State tokens (s0 to s15 = 16 states)
    for i in range(16):
        vocab[f's{i}'] = token_id
        token_id += 1
    
    # Action tokens (0 to 15 = 16 actions)  
    for i in range(16):
        vocab[f'a{i}'] = token_id
        token_id += 1
    
    return vocab


def tokenize_sequence(
    execution_path: List[Tuple[str, int, str]], 
    vocab: Dict[str, int]
) -> List[int]:
    """
    Convert FSM execution path to tokenized sequence.
    
    Each (state, action, next_state) triple becomes [state_token, action_token]
    following the (S, A) pairs format specified in plan.md.
    
    Args:
        execution_path: List of (current_state, action, next_state) tuples
        vocab: Vocabulary mapping from tokens to IDs
        
    Returns:
        List of token IDs representing the sequence
        
    Example:
        [('s0', 1, 's1'), ('s1', 2, 's0')] -> [s0_token, a1_token, s1_token, a2_token]
    """
    tokens = []
    
    for state, action, next_state in execution_path:
        state_token = vocab.get(state, vocab['<PAD>'])
        action_token = vocab.get(f'a{action}', vocab['<PAD>'])
        
        tokens.extend([state_token, action_token])
    
    return tokens


def generate_sample(
    fsm_id: int,
    vocab: Dict[str, int],
    sequence_length: int = 64,
    context_length: int = 256,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate one training/test sample.
    
    Returns:
        Dictionary with tokenized sequence, original data, and metadata
    """
    if seed is not None:
        set_seed(seed + fsm_id)  # Ensure each FSM gets unique seed
    
    # Generate random FSM
    fsm = generate_random_fsms(1, seed=seed + fsm_id if seed else None)[0]
    
    # Generate truncated sequence with random mode and amount
    actions, original_path, truncated_path, truncation_info = generate_truncated_sequence(
        fsm=fsm,
        sequence_length=sequence_length,
        truncate_mode="random",  # Use random for variety
        truncate_amount=None,     # Let it choose 0-3 randomly
        seed=seed + fsm_id if seed else None
    )
    
    # Tokenize the truncated sequence
    tokens = tokenize_sequence(truncated_path, vocab)
    
    # Add BOS token at the beginning
    tokens = [vocab['<BOS>']] + tokens
    
    # Pad or truncate to context length
    if len(tokens) > context_length:
        tokens = tokens[:context_length]
    else:
        tokens.extend([vocab['<PAD>']] * (context_length - len(tokens)))
    
    return {
        'tokens': tokens,
        'fsm_id': fsm_id,
        'fsm': fsm,  # Store original FSM for analysis
        'original_path': original_path,
        'truncated_path': truncated_path,
        'actions': actions,
        'truncation_info': truncation_info,
        'sequence_length': len(truncated_path),
        'vocab_size': len(vocab)
    }


def generate_dataset_split(
    name: str,
    num_samples: int,
    vocab: Dict[str, int],
    sequence_length: int,
    context_length: int,
    base_seed: int,
    output_dir: Path,
    output_format: str = 'pkl'
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Generate one split of the dataset."""
    
    print(f"Generating {name} split with {num_samples} samples...")
    
    samples = []
    metadata = {
        'split': name,
        'num_samples': num_samples,
        'sequence_length': sequence_length, 
        'context_length': context_length,
        'vocab_size': len(vocab),
        'base_seed': base_seed
    }
    
    # Generate samples with progress bar
    for i in tqdm(range(num_samples), desc=f"Generating {name}"):
        sample = generate_sample(
            fsm_id=i,
            vocab=vocab,
            sequence_length=sequence_length,
            context_length=context_length,
            seed=base_seed + i
        )
        samples.append(sample)
    
    # Save the split in specified format
    if output_format == 'pkl':
        split_data = save_dataset_pkl(samples, metadata, vocab, output_dir, name)
        print(f"Saved {name} split to {output_dir}/{name}_dataset.pkl")
    elif output_format == 'json':
        split_data = save_dataset_json(samples, metadata, vocab, output_dir, name)
        print(f"Saved {name} split to {output_dir}/{name}_tokens.json and {name}_raw.json")
    elif output_format == 'parquet':
        split_data = save_dataset_parquet(samples, metadata, vocab, output_dir, name)
        print(f"Saved {name} split to {output_dir}/{name}_tokens.parquet and {name}_full.parquet")
    else:
        # For HDF5, we'll handle it differently since it's one file for all splits
        split_data = {'samples': samples, 'metadata': metadata, 'vocab': vocab}
    
    # Save human-readable summary (always JSON)
    summary_file = output_dir / f'{name}_summary.json'
    summary = {
        'metadata': metadata,
        'sample_truncation_modes': {},
        'sample_sequence_lengths': {}
    }
    
    # Analyze truncation mode distribution  
    raw_counts = {}
    for sample in samples:
        info = sample['truncation_info']
        prefix = info.split('(')[0]
        raw_counts[prefix] = raw_counts.get(prefix, 0) + 1
    
    # Group them logically
    grouped_counts = {
        'start_state': raw_counts.get('start_state_s0', 0) + raw_counts.get('start_state_fallback', 0),
        'action': raw_counts.get('action_start', 0),
        'non_start_state': raw_counts.get('non_start_state', 0) + raw_counts.get('non_start_via_transition', 0)
    }
    summary['sample_truncation_modes'] = grouped_counts
    
    # Count sequence lengths
    for sample in samples:
        seq_len = sample['sequence_length']
        summary['sample_sequence_lengths'][seq_len] = summary['sample_sequence_lengths'].get(seq_len, 0) + 1
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return samples, metadata


def main():
    """Main dataset generation entry point."""
    parser = argparse.ArgumentParser(description='Generate FSM dataset')
    parser.add_argument('--output-dir', '-o', default='./data',
                       help='Output directory for dataset files')
    parser.add_argument('--train-samples', type=int, default=4000,
                       help='Number of training samples (4000 as per plan)')
    parser.add_argument('--val-samples', type=int, default=1000,  
                       help='Number of validation samples (1000 as per plan)')
    parser.add_argument('--test-samples', type=int, default=1000,
                       help='Number of test samples (1000 as per plan)') 
    parser.add_argument('--sequence-length', type=int, default=64,
                       help='Target sequence length before truncation')
    parser.add_argument('--context-length', type=int, default=256,
                       help='Maximum context length for tokenized sequences')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed')
    parser.add_argument('--format', choices=['pkl', 'json', 'parquet', 'hdf5'], 
                       default='pkl', help='Output format for dataset')
    parser.add_argument('--vocab-size', type=int, default=36,
                       help='Vocabulary size (should be 36 for current setup)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating dataset with:")
    print(f"  Training samples: {args.train_samples}")
    print(f"  Validation samples: {args.val_samples}")
    print(f"  Test samples: {args.test_samples}")
    print(f"  Sequence length: {args.sequence_length}")
    print(f"  Context length: {args.context_length}")
    print(f"  Base seed: {args.seed}")
    print(f"  Output format: {args.format}")
    print(f"  Output directory: {output_dir}")
    
    # Create vocabulary
    print("Creating vocabulary...")
    vocab = create_vocabulary_mapping()
    actual_vocab_size = len(vocab)
    print(f"Created vocabulary with {actual_vocab_size} tokens")
    
    if args.vocab_size != actual_vocab_size:
        print(f"Warning: Requested vocab size {args.vocab_size} != actual size {actual_vocab_size}")
    
    # Save vocabulary
    vocab_file = output_dir / 'vocab.json'
    with open(vocab_file, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocabulary to {vocab_file}")
    
    # Generate dataset splits
    set_seed(args.seed)
    
    splits = [
        ('train', args.train_samples, args.seed),
        ('val', args.val_samples, args.seed + 10000),
        ('test', args.test_samples, args.seed + 20000)
    ]
    
    all_metadata = {}
    all_splits_data = {}  # For HDF5 format
    
    for split_name, num_samples, split_seed in splits:
        samples, metadata = generate_dataset_split(
            name=split_name,
            num_samples=num_samples,
            vocab=vocab,
            sequence_length=args.sequence_length,
            context_length=args.context_length,
            base_seed=split_seed,
            output_dir=output_dir,
            output_format=args.format
        )
        all_metadata[split_name] = metadata
        all_splits_data[split_name] = (samples, metadata)
    
    # For HDF5, save all splits in one file
    if args.format == 'hdf5':
        save_dataset_hdf5(all_splits_data, vocab, output_dir)
        print(f"Saved all splits to {output_dir}/dataset.h5")
    
    # Save overall metadata
    dataset_info = {
        'generation_args': vars(args),
        'splits': all_metadata,
        'vocab_size': actual_vocab_size,
        'total_samples': args.train_samples + args.val_samples + args.test_samples
    }
    
    info_file = output_dir / 'dataset_info.json'
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n✅ Dataset generation complete!")
    print(f"Files created in {output_dir}:")
    for file_path in sorted(output_dir.glob('*')):
        print(f"  {file_path.name}")


if __name__ == '__main__':
    main()