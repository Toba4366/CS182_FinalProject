"""
PyTorch dataset for Moore machine in-context learning.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional
import json
import pickle
from pathlib import Path

from ..fsm.moore_machine import MooreMachine, MooreMachineGenerator
from ..fsm.fsm_utils import create_input_output_examples, format_sequence_for_model


class MooreMachineDataset(Dataset):
    """
    PyTorch dataset for Moore machine in-context learning tasks.
    
    Each sample contains:
    - A prompt with demonstration input-output examples
    - A test input sequence 
    - The expected output sequence
    """
    
    def __init__(self,
                 num_machines: int = 1000,
                 examples_per_machine: int = 5,
                 sequence_length: int = 10,
                 test_sequence_length: int = 15,
                 vocab_size: int = 100,
                 seed: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            num_machines: Number of different Moore machines to generate
            examples_per_machine: Number of demo examples per machine
            sequence_length: Length of each demonstration sequence
            test_sequence_length: Length of test sequence
            vocab_size: Size of vocabulary for tokenization
            seed: Random seed for reproducibility
        """
        self.num_machines = num_machines
        self.examples_per_machine = examples_per_machine
        self.sequence_length = sequence_length
        self.test_sequence_length = test_sequence_length
        self.vocab_size = vocab_size
        self.seed = seed
        
        # Initialize generator
        self.generator = MooreMachineGenerator(seed=seed)
        
        # Build vocabulary
        self.vocab = self._build_vocabulary()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        
        # Generate all samples
        self.samples = self._generate_samples()
    
    def _build_vocabulary(self) -> List[str]:
        """Build vocabulary for tokenization."""
        vocab = ["<pad>", "<sep>", "<start>", "<end>"]
        
        # Action tokens (A0, A1, ..., A7)
        vocab.extend([f"A{i}" for i in range(8)])
        
        # Output tokens (out_0, out_1, out_2)
        vocab.extend(["out_0", "out_1", "out_2"])
        
        # Pad to vocab_size if needed
        while len(vocab) < self.vocab_size:
            vocab.append(f"<unk_{len(vocab)}>")
        
        return vocab[:self.vocab_size]
    
    def _tokenize_sequence(self, sequence: str) -> List[int]:
        """Convert sequence string to token IDs."""
        tokens = sequence.split()
        token_ids = []
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.token_to_id.get("<unk>", 0))
        return token_ids
    
    def _generate_samples(self) -> List[Dict]:
        """Generate all dataset samples."""
        samples = []
        
        for i in range(self.num_machines):
            # Generate random Moore machine
            machine_seed = None if self.seed is None else self.seed + i
            fsm = self.generator.generate()
            
            # Create demonstration examples
            demo_examples = create_input_output_examples(
                fsm, 
                num_examples=self.examples_per_machine,
                sequence_length=self.sequence_length,
                seed=machine_seed
            )
            
            # Format demonstration prompt
            demo_parts = []
            for actions, outputs in demo_examples:
                demo_seq = format_sequence_for_model(actions, outputs)
                demo_parts.append(demo_seq)
            
            prompt = " <sep> ".join(demo_parts)
            
            # Generate test sequence
            test_example = create_input_output_examples(
                fsm,
                num_examples=1,
                sequence_length=self.test_sequence_length,
                seed=machine_seed + 1000  # Different seed for test
            )[0]
            test_actions, test_outputs = test_example
            
            test_input = " ".join([f"A{a}" for a in test_actions])
            test_output = " ".join(test_outputs)
            
            sample = {
                'prompt': prompt,
                'test_input': test_input,
                'test_output': test_output,
                'machine_id': i,
                'fsm': fsm.to_dict()
            }
            samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Tokenize sequences
        prompt_tokens = self._tokenize_sequence(sample['prompt'])
        test_input_tokens = self._tokenize_sequence(sample['test_input'])
        test_output_tokens = self._tokenize_sequence(sample['test_output'])
        
        # Create full sequence: prompt + separator + test_input + separator + test_output + end
        full_sequence = (prompt_tokens + 
                        [self.token_to_id['<sep>']] + 
                        test_input_tokens +
                        [self.token_to_id['<sep>']] +
                        test_output_tokens +
                        [self.token_to_id['<end>']])
        
        # Create input (all but last token) and target (all but first token)
        input_ids = full_sequence[:-1]
        target_ids = full_sequence[1:]
        
        # Create labels with masking: only predict the test output part
        labels = [-100] * len(target_ids)  # Start with all masked
        # Find where test output starts (after second <sep>)
        sep_count = 0
        output_start_idx = 0
        for i, token_id in enumerate(input_ids):
            if token_id == self.token_to_id['<sep>']:
                sep_count += 1
                if sep_count == 2:
                    output_start_idx = i + 1
                    break
        
        # Unmask the test output tokens for loss computation
        for i in range(output_start_idx, len(labels)):
            labels[i] = target_ids[i]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(labels, dtype=torch.long),
            'machine_id': sample['machine_id']
        }
    
    def save_to_disk(self, filepath: str):
        """Save dataset to disk."""
        data = {
            'samples': self.samples,
            'vocab': self.vocab,
            'token_to_id': self.token_to_id,
            'config': {
                'num_machines': self.num_machines,
                'examples_per_machine': self.examples_per_machine,
                'sequence_length': self.sequence_length,
                'test_sequence_length': self.test_sequence_length,
                'vocab_size': self.vocab_size,
                'seed': self.seed
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load_from_disk(cls, filepath: str) -> "MooreMachineDataset":
        """Load dataset from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create new dataset instance
        dataset = cls(**data['config'])
        dataset.samples = data['samples']
        dataset.vocab = data['vocab']
        dataset.token_to_id = data['token_to_id']
        dataset.id_to_token = {i: token for token, i in dataset.token_to_id.items()}
        
        return dataset


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    Pads sequences to the same length within a batch.
    """
    # Find max lengths
    max_input_len = max(len(item['input_ids']) for item in batch)
    max_target_len = max(len(item['target_ids']) for item in batch)
    
    # Pad sequences
    input_ids = []
    target_ids = []
    attention_masks = []
    machine_ids = []
    
    for item in batch:
        # Pad input
        input_len = len(item['input_ids'])
        padded_input = torch.cat([
            item['input_ids'],
            torch.zeros(max_input_len - input_len, dtype=torch.long)
        ])
        input_ids.append(padded_input)
        
        # Create attention mask
        attention_mask = torch.cat([
            torch.ones(input_len, dtype=torch.bool),
            torch.zeros(max_input_len - input_len, dtype=torch.bool)
        ])
        attention_masks.append(attention_mask)
        
        # Pad target
        target_len = len(item['target_ids'])
        padded_target = torch.cat([
            item['target_ids'],
            torch.full((max_target_len - target_len,), -100, dtype=torch.long)  # -100 is ignored by loss
        ])
        target_ids.append(padded_target)
        
        machine_ids.append(item['machine_id'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'target_ids': torch.stack(target_ids),
        'machine_ids': torch.tensor(machine_ids, dtype=torch.long)
    }