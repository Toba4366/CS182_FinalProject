"""
Evaluation utilities for ICL performance assessment.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader

from ..training.models import SimpleTransformer
from ..training.dataset import MooreMachineDataset, collate_fn
from ..fsm.fsm_utils import compute_sequence_accuracy


def evaluate_icl_performance(model: SimpleTransformer,
                            dataset: MooreMachineDataset,
                            device: str = 'cpu',
                            batch_size: int = 16,
                            max_generate_length: int = 50) -> Dict[str, float]:
    """
    Evaluate in-context learning performance on a dataset.
    
    Args:
        model: Trained transformer model
        dataset: Evaluation dataset
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        max_generate_length: Maximum length to generate
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model = model.to(device)
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    all_accuracies = []
    all_exact_matches = []
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Compute loss
            logits = model(input_ids, attention_mask)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., :].contiguous()
            
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            loss = torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, ignore_index=-100, reduction='mean'
            )
            
            batch_size_actual = input_ids.size(0)
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual
            
            # Generate sequences for accuracy evaluation
            # Find separator token to split prompt from test input
            sep_token_id = dataset.token_to_id['<sep>']
            end_token_id = dataset.token_to_id['<end>']
            
            for i in range(batch_size_actual):
                # Find where the test sequence starts (after last separator)
                input_seq = input_ids[i].cpu().numpy()
                sep_positions = np.where(input_seq == sep_token_id)[0]
                
                if len(sep_positions) >= 2:
                    # Take input up to the last separator (before test output)
                    prompt_end = sep_positions[-1] + 1
                    prompt_input = input_ids[i:i+1, :prompt_end]
                    
                    # Generate continuation
                    generated = model.generate(
                        prompt_input,
                        max_length=prompt_end + max_generate_length,
                        temperature=1.0,
                        do_sample=False,  # Use greedy decoding for evaluation
                        eos_token_id=end_token_id
                    )
                    
                    # Extract generated sequence
                    generated_tokens = generated[0, prompt_end:].cpu().numpy()
                    
                    # Remove padding and end token
                    end_pos = np.where(generated_tokens == end_token_id)[0]
                    if len(end_pos) > 0:
                        generated_tokens = generated_tokens[:end_pos[0]]
                    
                    # Convert to strings
                    generated_outputs = []
                    target_outputs = []
                    
                    for token_id in generated_tokens:
                        if token_id in dataset.id_to_token:
                            token = dataset.id_to_token[token_id]
                            if token.startswith('out_'):
                                generated_outputs.append(token)
                    
                    # Extract target sequence (ignore -100 tokens)
                    target_seq = target_ids[i].cpu().numpy()
                    for token_id in target_seq:
                        if token_id != -100 and token_id in dataset.id_to_token:
                            token = dataset.id_to_token[token_id]
                            if token.startswith('out_'):
                                target_outputs.append(token)
                    
                    # Compute accuracy
                    accuracy = compute_sequence_accuracy(generated_outputs, target_outputs)
                    exact_match = 1.0 if accuracy == 1.0 else 0.0
                    
                    all_accuracies.append(accuracy)
                    all_exact_matches.append(exact_match)
    
    # Compute average metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    avg_accuracy = np.mean(all_accuracies) if all_accuracies else 0.0
    exact_match_rate = np.mean(all_exact_matches) if all_exact_matches else 0.0
    perplexity = np.exp(avg_loss)
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'accuracy': avg_accuracy,
        'exact_match_rate': exact_match_rate,
        'num_samples': len(all_accuracies)
    }


def compute_accuracy_metrics(predicted_sequences: List[List[str]],
                           target_sequences: List[List[str]]) -> Dict[str, float]:
    """
    Compute various accuracy metrics for sequence predictions.
    
    Args:
        predicted_sequences: List of predicted output sequences
        target_sequences: List of target output sequences
        
    Returns:
        Dictionary with accuracy metrics
    """
    if len(predicted_sequences) != len(target_sequences):
        raise ValueError("Number of predicted and target sequences must match")
    
    token_accuracies = []
    exact_matches = []
    prefix_matches = []
    
    for pred, target in zip(predicted_sequences, target_sequences):
        # Token-level accuracy
        token_acc = compute_sequence_accuracy(pred, target)
        token_accuracies.append(token_acc)
        
        # Exact match
        exact_match = 1.0 if pred == target else 0.0
        exact_matches.append(exact_match)
        
        # Prefix match (how much of the beginning matches)
        min_len = min(len(pred), len(target))
        prefix_len = 0
        for i in range(min_len):
            if pred[i] == target[i]:
                prefix_len += 1
            else:
                break
        
        prefix_acc = prefix_len / len(target) if len(target) > 0 else 1.0
        prefix_matches.append(prefix_acc)
    
    return {
        'token_accuracy': np.mean(token_accuracies),
        'exact_match_rate': np.mean(exact_matches),
        'prefix_accuracy': np.mean(prefix_matches),
        'token_accuracy_std': np.std(token_accuracies),
        'num_sequences': len(predicted_sequences)
    }


def analyze_performance_by_complexity(model: SimpleTransformer,
                                     dataset: MooreMachineDataset,
                                     device: str = 'cpu') -> Dict[str, List[float]]:
    """
    Analyze performance breakdown by FSM complexity metrics.
    
    Args:
        model: Trained model
        dataset: Evaluation dataset
        device: Device for evaluation
        
    Returns:
        Dictionary with performance metrics grouped by complexity
    """
    model.eval()
    model = model.to(device)
    
    # Group samples by FSM characteristics
    complexity_groups = {
        'num_actions': {},
        'num_transitions': {},
        'has_self_loops': {'yes': [], 'no': []}
    }
    
    for i, sample in enumerate(dataset.samples):
        fsm_dict = sample['fsm']
        
        # Number of actions
        num_actions = fsm_dict['num_actions']
        if num_actions not in complexity_groups['num_actions']:
            complexity_groups['num_actions'][num_actions] = []
        complexity_groups['num_actions'][num_actions].append(i)
        
        # Number of transitions
        num_transitions = len(fsm_dict['transitions'])
        if num_transitions not in complexity_groups['num_transitions']:
            complexity_groups['num_transitions'][num_transitions] = []
        complexity_groups['num_transitions'][num_transitions].append(i)
        
        # Self-loops
        has_self_loop = any(
            int(key.split(',')[0]) == value
            for key, value in fsm_dict['transitions'].items()
        )
        group_key = 'yes' if has_self_loop else 'no'
        complexity_groups['has_self_loops'][group_key].append(i)
    
    # Evaluate each group
    results = {}
    
    for category, groups in complexity_groups.items():
        results[category] = {}
        
        for group_name, indices in groups.items():
            if len(indices) == 0:
                continue
            
            # Create subset dataset
            subset_samples = [dataset.samples[i] for i in indices]
            
            # Create temporary dataset for evaluation
            subset_dataset = MooreMachineDataset.__new__(MooreMachineDataset)
            subset_dataset.samples = subset_samples
            subset_dataset.vocab = dataset.vocab
            subset_dataset.token_to_id = dataset.token_to_id
            subset_dataset.id_to_token = dataset.id_to_token
            
            # Evaluate
            metrics = evaluate_icl_performance(model, subset_dataset, device)
            results[category][str(group_name)] = metrics
    
    return results