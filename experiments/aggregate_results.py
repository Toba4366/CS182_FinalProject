"""
Aggregate results from multiple runs.

Computes mean and std for test accuracy across runs.
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np


def aggregate_results(base_dir="results"):
    """Aggregate all experimental results."""
    base_path = Path(base_dir)
    
    # Group results by experiment
    experiments = defaultdict(list)
    
    # Find all result files
    for result_file in base_path.rglob("run_*.json"):
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        exp_name = result.get('experiment_name', 'unknown')
        experiments[exp_name].append(result)
    
    # Compute statistics
    print("=" * 80)
    print("Experiment Results Summary")
    print("=" * 80)
    print()
    
    summary = {}
    
    for exp_name, results in sorted(experiments.items()):
        test_accs = [r['test_results']['accuracy'] for r in results]
        test_losses = [r['test_results']['loss'] for r in results]
        
        summary[exp_name] = {
            'num_runs': len(results),
            'test_accuracy': {
                'mean': float(np.mean(test_accs)),
                'std': float(np.std(test_accs)),
                'min': float(np.min(test_accs)),
                'max': float(np.max(test_accs)),
            },
            'test_loss': {
                'mean': float(np.mean(test_losses)),
                'std': float(np.std(test_losses)),
            }
        }
        
        print(f"{exp_name}:")
        print(f"  Runs: {len(results)}")
        print(f"  Test Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
        print(f"  Test Loss: {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}")
        print()
    
    # Save summary
    summary_file = base_path / "aggregated_results.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    aggregate_results()
