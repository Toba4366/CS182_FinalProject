#!/usr/bin/env python3
"""
FSM Dataset Format Conversion Utilities

This module provides functions to convert FSM datasets between different file formats
and save datasets directly during generation. Supports 4 formats optimized for different use cases:

1. PKL (Pickle): Fastest loading in Python, smallest file size for training
2. JSON: Human-readable, debuggable, cross-language compatible  
3. Parquet: Industry standard, compressed, works with data analysis tools
4. HDF5: Scientific computing standard, efficient for very large datasets

Key Features:
- Format-agnostic dataset generation and saving
- Cross-format data integrity preservation  
- Optimized storage for different access patterns
- Comprehensive metadata preservation

Usage:
    # Convert existing pickle dataset to other formats
    python utils/convert_dataset.py --input-dir data/full_dataset_pkl --output-dir data/full_dataset_json
    
    # Or generate datasets directly in preferred format using generate_dataset.py:
    python utils/generate_dataset.py --format json
"""

import json
import pickle
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, Any, List
import argparse


def convert_pkl_to_parquet(input_dir: Path, output_dir: Path):
    """Convert pickle dataset to Parquet format."""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting dataset from {input_dir} to Parquet format in {output_dir}")
    
    # Copy JSON files
    for json_file in ['vocab.json', 'dataset_info.json']:
        src = input_dir / json_file
        if src.exists():
            import shutil
            shutil.copy2(src, output_dir / json_file)
            print(f"✓ Copied {json_file}")
    
    # Convert each split
    for split in ['train', 'val', 'test']:
        pkl_file = input_dir / f'{split}_dataset.pkl'
        if not pkl_file.exists():
            continue
            
        print(f"Converting {split} split to Parquet...")
        
        # Load pickle data
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        samples = data['samples']
        
        # Create DataFrame for main training data
        df_data = []
        for i, sample in enumerate(samples):
            df_data.append({
                'id': i,
                'tokens': sample['tokens'],  # Store as list
                'sequence_length': sample['sequence_length'],
                'truncation_mode': sample['truncation_info'].split('(')[0],
                'fsm_id': sample['fsm_id'],
                'truncation_info': sample['truncation_info']
            })
        
        df = pd.DataFrame(df_data)
        
        # Save main tokens as Parquet
        tokens_file = output_dir / f'{split}_tokens.parquet'
        df[['id', 'tokens', 'sequence_length', 'truncation_mode']].to_parquet(tokens_file, index=False)
        print(f"  ✓ Saved {tokens_file}")
        
        # Save full data as Parquet
        # For complex data like FSM dicts, we'll store as JSON strings
        df_full = df.copy()
        for i, sample in enumerate(samples):
            df_full.loc[i, 'fsm'] = json.dumps(sample['fsm'])
            df_full.loc[i, 'original_path'] = json.dumps(sample['original_path'])
            df_full.loc[i, 'truncated_path'] = json.dumps(sample['truncated_path'])
            df_full.loc[i, 'actions'] = json.dumps(sample['actions'])
        
        full_file = output_dir / f'{split}_full.parquet'
        df_full.to_parquet(full_file, index=False)
        print(f"  ✓ Saved {full_file}")
        
        # Copy summary
        summary_file = input_dir / f'{split}_summary.json'
        if summary_file.exists():
            import shutil
            shutil.copy2(summary_file, output_dir / f'{split}_summary.json')


def convert_pkl_to_hdf5(input_dir: Path, output_dir: Path):
    """Convert pickle dataset to HDF5 format."""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting dataset from {input_dir} to HDF5 format in {output_dir}")
    
    # Copy JSON files
    for json_file in ['vocab.json', 'dataset_info.json']:
        src = input_dir / json_file
        if src.exists():
            import shutil
            shutil.copy2(src, output_dir / json_file)
            print(f"✓ Copied {json_file}")
    
    # Create single HDF5 file for all splits
    hdf5_file = output_dir / 'dataset.h5'
    
    with h5py.File(hdf5_file, 'w') as h5f:
        # Convert each split
        for split in ['train', 'val', 'test']:
            pkl_file = input_dir / f'{split}_dataset.pkl'
            if not pkl_file.exists():
                continue
                
            print(f"Converting {split} split to HDF5...")
            
            # Load pickle data
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            samples = data['samples']
            
            # Create group for this split
            split_group = h5f.create_group(split)
            
            # Store metadata as attributes
            for key, value in data['metadata'].items():
                split_group.attrs[key] = str(value)
            
            # Prepare data arrays
            num_samples = len(samples)
            max_token_length = max(len(sample['tokens']) for sample in samples)
            max_path_length = max(len(sample['truncated_path']) for sample in samples)
            
            # Create datasets
            tokens_ds = split_group.create_dataset('tokens', (num_samples, max_token_length), dtype='i4', fillvalue=0)
            seq_len_ds = split_group.create_dataset('sequence_lengths', (num_samples,), dtype='i4')
            fsm_id_ds = split_group.create_dataset('fsm_ids', (num_samples,), dtype='i4')
            
            # Variable length string datasets
            dt = h5py.special_dtype(vlen=str)
            trunc_mode_ds = split_group.create_dataset('truncation_modes', (num_samples,), dtype=dt)
            trunc_info_ds = split_group.create_dataset('truncation_info', (num_samples,), dtype=dt)
            fsm_ds = split_group.create_dataset('fsms', (num_samples,), dtype=dt)
            actions_ds = split_group.create_dataset('actions', (num_samples,), dtype=dt)
            
            # Fill datasets
            for i, sample in enumerate(samples):
                # Pad tokens to max length
                tokens = sample['tokens'][:max_token_length]
                if len(tokens) < max_token_length:
                    tokens.extend([0] * (max_token_length - len(tokens)))
                
                tokens_ds[i] = tokens
                seq_len_ds[i] = sample['sequence_length']
                fsm_id_ds[i] = sample['fsm_id']
                trunc_mode_ds[i] = sample['truncation_info'].split('(')[0]
                trunc_info_ds[i] = sample['truncation_info']
                fsm_ds[i] = json.dumps(sample['fsm'])
                actions_ds[i] = json.dumps(sample['actions'])
            
            print(f"  ✓ Added {split} group with {num_samples} samples")
            
            # Copy summary
            summary_file = input_dir / f'{split}_summary.json'
            if summary_file.exists():
                import shutil
                shutil.copy2(summary_file, output_dir / f'{split}_summary.json')
    
    print(f"  ✓ Saved HDF5 file: {hdf5_file}")


def save_dataset_pkl(samples: List[Dict], metadata: Dict, vocab: Dict, output_dir: Path, split_name: str):
    """Save dataset in pickle format."""
    split_data = {
        'samples': samples,
        'metadata': metadata,
        'vocab': vocab
    }
    
    output_file = output_dir / f'{split_name}_dataset.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(split_data, f)
    
    return split_data


def save_dataset_json(samples: List[Dict], metadata: Dict, vocab: Dict, output_dir: Path, split_name: str):
    """Save dataset in JSON format."""
    # Main tokens file
    tokens_data = {
        'metadata': metadata,
        'samples': []
    }
    
    # Raw data file  
    raw_data = {
        'metadata': metadata,
        'samples': []
    }
    
    for i, sample in enumerate(samples):
        tokens_data['samples'].append({
            'id': i,
            'tokens': sample['tokens'],
            'sequence_length': sample['sequence_length'],
            'truncation_mode': sample['truncation_info'].split('(')[0]
        })
        
        raw_data['samples'].append(sample)
    
    # Save files
    tokens_file = output_dir / f'{split_name}_tokens.json'
    with open(tokens_file, 'w') as f:
        json.dump(tokens_data, f, indent=2)
    
    raw_file = output_dir / f'{split_name}_raw.json'
    with open(raw_file, 'w') as f:
        json.dump(raw_data, f, indent=2)
    
    return {'tokens': tokens_data, 'raw': raw_data}


def save_dataset_parquet(samples: List[Dict], metadata: Dict, vocab: Dict, output_dir: Path, split_name: str):
    """Save dataset in Parquet format."""
    # Create DataFrame
    df_data = []
    for i, sample in enumerate(samples):
        df_row = {
            'id': i,
            'tokens': sample['tokens'],
            'sequence_length': sample['sequence_length'],
            'truncation_mode': sample['truncation_info'].split('(')[0],
            'fsm_id': sample['fsm_id'],
            'truncation_info': sample['truncation_info'],
            'fsm': json.dumps(sample['fsm']),
            'original_path': json.dumps(sample['original_path']),
            'truncated_path': json.dumps(sample['truncated_path']),
            'actions': json.dumps(sample['actions'])
        }
        df_data.append(df_row)
    
    df = pd.DataFrame(df_data)
    
    # Save tokens file
    tokens_file = output_dir / f'{split_name}_tokens.parquet'
    df[['id', 'tokens', 'sequence_length', 'truncation_mode']].to_parquet(tokens_file, index=False)
    
    # Save full file
    full_file = output_dir / f'{split_name}_full.parquet'
    df.to_parquet(full_file, index=False)
    
    return df


def save_dataset_hdf5(all_splits_data: Dict, vocab: Dict, output_dir: Path):
    """Save all splits in a single HDF5 file."""
    hdf5_file = output_dir / 'dataset.h5'
    
    with h5py.File(hdf5_file, 'w') as h5f:
        # Store vocab as attribute
        h5f.attrs['vocab'] = json.dumps(vocab)
        
        for split_name, (samples, metadata) in all_splits_data.items():
            # Create group
            split_group = h5f.create_group(split_name)
            
            # Store metadata
            for key, value in metadata.items():
                split_group.attrs[key] = str(value)
            
            # Prepare arrays
            num_samples = len(samples)
            if num_samples == 0:
                continue
                
            max_token_length = max(len(sample['tokens']) for sample in samples)
            
            # Create datasets
            tokens_ds = split_group.create_dataset('tokens', (num_samples, max_token_length), dtype='i4', fillvalue=0)
            seq_len_ds = split_group.create_dataset('sequence_lengths', (num_samples,), dtype='i4')
            fsm_id_ds = split_group.create_dataset('fsm_ids', (num_samples,), dtype='i4')
            
            # String datasets
            dt = h5py.special_dtype(vlen=str)
            trunc_mode_ds = split_group.create_dataset('truncation_modes', (num_samples,), dtype=dt)
            trunc_info_ds = split_group.create_dataset('truncation_info', (num_samples,), dtype=dt)
            fsm_ds = split_group.create_dataset('fsms', (num_samples,), dtype=dt)
            actions_ds = split_group.create_dataset('actions', (num_samples,), dtype=dt)
            
            # Fill data
            for i, sample in enumerate(samples):
                tokens = sample['tokens'][:max_token_length]
                if len(tokens) < max_token_length:
                    tokens.extend([0] * (max_token_length - len(tokens)))
                
                tokens_ds[i] = tokens
                seq_len_ds[i] = sample['sequence_length']
                fsm_id_ds[i] = sample['fsm_id']
                trunc_mode_ds[i] = sample['truncation_info'].split('(')[0]
                trunc_info_ds[i] = sample['truncation_info']
                fsm_ds[i] = json.dumps(sample['fsm'])
                actions_ds[i] = json.dumps(sample['actions'])


def convert_pkl_to_json(input_dir: Path, output_dir: Path):
    """Convert pickle dataset to JSON format."""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting dataset from {input_dir} to {output_dir}")
    
    # Copy vocabulary (already JSON)
    vocab_file = input_dir / 'vocab.json'
    if vocab_file.exists():
        import shutil
        shutil.copy2(vocab_file, output_dir / 'vocab.json')
        print("✓ Copied vocab.json")
    
    # Copy dataset info (already JSON)
    info_file = input_dir / 'dataset_info.json'
    if info_file.exists():
        import shutil
        shutil.copy2(info_file, output_dir / 'dataset_info.json')
        print("✓ Copied dataset_info.json")
    
    # Convert each split
    for split in ['train', 'val', 'test']:
        pkl_file = input_dir / f'{split}_dataset.pkl'
        if not pkl_file.exists():
            print(f"⚠️  {pkl_file} not found, skipping")
            continue
            
        print(f"Converting {split} split...")
        
        # Load pickle data
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract components
        samples = data['samples']
        metadata = data['metadata']
        
        # Create separate files for better organization
        
        # 1. Tokens file (the main training data)
        tokens_data = {
            'metadata': metadata,
            'samples': []
        }
        
        # 2. Raw data file (for analysis and debugging)
        raw_data = {
            'metadata': metadata, 
            'samples': []
        }
        
        for i, sample in enumerate(samples):
            # Main tokens file - just what you need for training
            tokens_data['samples'].append({
                'id': i,
                'tokens': sample['tokens'],
                'sequence_length': sample['sequence_length'],
                'truncation_mode': sample['truncation_info'].split('(')[0]
            })
            
            # Raw data file - everything for analysis
            raw_sample = {
                'id': i,
                'tokens': sample['tokens'],
                'fsm_id': sample['fsm_id'],
                'fsm': sample['fsm'],  # The actual FSM structure
                'original_path': sample['original_path'],
                'truncated_path': sample['truncated_path'],
                'actions': sample['actions'],
                'truncation_info': sample['truncation_info'],
                'sequence_length': sample['sequence_length']
            }
            raw_data['samples'].append(raw_sample)
        
        # Save tokens file (main training data)
        tokens_file = output_dir / f'{split}_tokens.json'
        with open(tokens_file, 'w') as f:
            json.dump(tokens_data, f, indent=2)
        print(f"  ✓ Saved {tokens_file} ({len(tokens_data['samples'])} samples)")
        
        # Save raw data file (for analysis)
        raw_file = output_dir / f'{split}_raw.json'
        with open(raw_file, 'w') as f:
            json.dump(raw_data, f, indent=2)
        print(f"  ✓ Saved {raw_file}")
        
        # Copy summary file
        summary_file = input_dir / f'{split}_summary.json'
        if summary_file.exists():
            import shutil
            shutil.copy2(summary_file, output_dir / f'{split}_summary.json')
            print(f"  ✓ Copied {split}_summary.json")
    
    print(f"\n✅ Conversion complete!")
    print(f"Files saved to: {output_dir}")
    
    # Show file sizes for comparison
    print(f"\nFile size comparison:")
    for split in ['train', 'val', 'test']:
        pkl_file = input_dir / f'{split}_dataset.pkl'
        json_file = output_dir / f'{split}_tokens.json'
        raw_file = output_dir / f'{split}_raw.json'
        
        if pkl_file.exists() and json_file.exists():
            pkl_size = pkl_file.stat().st_size / (1024*1024)  # MB
            json_size = json_file.stat().st_size / (1024*1024)  # MB
            raw_size = raw_file.stat().st_size / (1024*1024)  # MB
            
            print(f"  {split}: PKL={pkl_size:.1f}MB → JSON={json_size:.1f}MB + Raw={raw_size:.1f}MB")


def create_data_loader_example(output_dir: Path):
    """Create example code for loading the JSON dataset."""
    
    example_code = '''#!/usr/bin/env python3
"""
Example data loader for JSON-format FSM dataset.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any


class FSMDataset(Dataset):
    """Dataset class for JSON-format FSM data."""
    
    def __init__(self, tokens_file: str, max_length: int = 256):
        """
        Args:
            tokens_file: Path to {split}_tokens.json file
            max_length: Maximum sequence length (for padding/truncating)
        """
        with open(tokens_file, 'r') as f:
            data = json.load(f)
        
        self.metadata = data['metadata']
        self.samples = data['samples']
        self.max_length = max_length
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"Vocab size: {self.metadata['vocab_size']}")
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        tokens = sample['tokens'][:self.max_length]  # Truncate if needed
        
        # Pad to max_length 
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))  # Assuming 0 is PAD
        
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'sample_id': sample['id'],
            'truncation_mode': sample['truncation_mode']
        }


# Example usage:
if __name__ == '__main__':
    # Load training data
    dataset = FSMDataset('./data/full_dataset_json/train_tokens.json')
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=4
    )
    
    # Test loading a batch
    for batch in dataloader:
        print(f"Batch shape: {batch['input_ids'].shape}")
        print(f"Sample truncation modes: {batch['truncation_mode']}")
        break
'''
    
    example_file = output_dir / 'load_dataset_example.py'
    with open(example_file, 'w') as f:
        f.write(example_code)
    
    print(f"✓ Created example data loader: {example_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert pickle dataset to JSON')
    parser.add_argument('--input-dir', default='./data/full_dataset',
                       help='Input directory with pickle files')
    parser.add_argument('--output-dir', default='./data/full_dataset_json',
                       help='Output directory for JSON files')
    parser.add_argument('--create-example', action='store_true',
                       help='Create example data loading code')
    
    args = parser.parse_args()
    
    # Convert dataset
    convert_pkl_to_json(args.input_dir, args.output_dir)
    
    # Create example code
    if args.create_example:
        create_data_loader_example(Path(args.output_dir))


if __name__ == '__main__':
    main()