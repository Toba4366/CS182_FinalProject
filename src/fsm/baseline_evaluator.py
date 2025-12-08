"""
Baseline evaluator for Moore machine datasets.

This module provides utilities to evaluate the deterministic solver
on the ICL dataset and compute baseline accuracies.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, cast
import torch  # type: ignore

from .deterministic_solver import DeterministicSolver
from .trajectory_sampler import Trajectory


def evaluate_deterministic_baseline(
    samples: List[Dict[str, object]],
    sample_indices: List[int],
    num_demos: int | None = None,
) -> Dict[str, float]:
    """
    Evaluate the deterministic solver on a set of samples.
    
    Args:
        samples: List of all samples from the dataset
        sample_indices: Indices of samples to evaluate
        num_demos: Number of demos to use per sample. If None, uses all demos.
                   If specified, randomly samples that many demos (matching model training).
    
    Returns:
        Dictionary with:
        - "accuracy": Overall accuracy across all queries
        - "num_samples": Number of samples evaluated
        - "num_correct": Number of queries with 100% accuracy
    """
    import random
    solver = DeterministicSolver()
    
    total_accuracy = 0.0
    num_samples = 0
    num_correct = 0
    
    rng = random.Random(42)  # Fixed seed for reproducibility
    
    num_demos_used = None
    for idx in sample_indices:
        sample = samples[idx]
        all_demos = cast(List[Trajectory], sample["demos"])
        query = cast(Trajectory, sample["query"])
        
        # Select demos: use all if num_demos is None, otherwise sample
        if num_demos is None:
            demos = all_demos
            if num_demos_used is None:
                num_demos_used = len(all_demos)
        else:
            # Sample without replacement, but take all if num_demos >= len(all_demos)
            num_to_use = min(num_demos, len(all_demos))
            demos = rng.sample(all_demos, num_to_use)
            if num_demos_used is None:
                num_demos_used = num_to_use
        
        accuracy = solver.evaluate_trajectory(demos, query)
        total_accuracy += accuracy
        num_samples += 1
        
        if accuracy == 1.0:
            num_correct += 1
    
    overall_accuracy = total_accuracy / num_samples if num_samples > 0 else 0.0
    
    return {
        "accuracy": overall_accuracy,
        "num_samples": num_samples,
        "num_correct": num_correct,
        "num_incorrect": num_samples - num_correct,
        "num_demos_used": num_demos_used if num_demos_used is not None else 0,
    }


def evaluate_per_trajectory(
    samples: List[Dict[str, object]],
    sample_indices: List[int],
    num_demos: int | None = None,
) -> List[Dict[str, float]]:
    """
    Evaluate the deterministic solver on each trajectory individually.
    
    Args:
        samples: List of all samples from the dataset
        sample_indices: Indices of samples to evaluate
        num_demos: Number of demos to use per sample. If None, uses all demos.
                   If specified, randomly samples that many demos (matching model training).
    
    Returns:
        List of dictionaries, one per sample, with:
        - "sample_idx": Index of the sample
        - "accuracy": Accuracy for this trajectory
    """
    import random
    solver = DeterministicSolver()
    results = []
    
    rng = random.Random(42)  # Fixed seed for reproducibility
    
    for idx in sample_indices:
        sample = samples[idx]
        all_demos = cast(List[Trajectory], sample["demos"])
        query = cast(Trajectory, sample["query"])
        
        # Select demos: use all if num_demos is None, otherwise sample
        if num_demos is None:
            demos = all_demos
        else:
            # Sample without replacement, but take all if num_demos >= len(all_demos)
            num_to_use = min(num_demos, len(all_demos))
            demos = rng.sample(all_demos, num_to_use)
        
        accuracy = solver.evaluate_trajectory(demos, query)
        
        results.append({
            "sample_idx": idx,
            "accuracy": accuracy,
        })
    
    return results

