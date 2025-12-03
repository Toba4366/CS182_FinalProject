#!/usr/bin/env python3
"""
Run All RNN Exploration Experiments
====================================

This script orchestrates all the RNN improvement experiments:
1. Capacity tests (d_model: 256, 512, 1024)
2. Depth tests (num_layers: 2, 5, 16)
3. GRU baseline comparison

Usage:
    python experiments/run_all_exploration.py --mode all
    python experiments/run_all_exploration.py --mode capacity
    python experiments/run_all_exploration.py --mode depth
    python experiments/run_all_exploration.py --mode gru
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print status."""
    print("\n" + "=" * 80)
    print(f"🚀 {description}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed with code {result.returncode}")
        return False
    else:
        print(f"\n✅ SUCCESS: {description} completed")
        return True


def run_capacity_experiments():
    """Run capacity experiments (varying hidden dimension)."""
    experiments = [
        (256, "rnn_d256_baseline", "Baseline RNN (256 hidden units)"),
        (512, "rnn_d512", "Large RNN (512 hidden units)"),
        (1024, "rnn_d1024", "Extra Large RNN (1024 hidden units)"),
    ]
    
    results = []
    for d_model, exp_name, desc in experiments:
        cmd = [
            sys.executable,
            "experiments/explorations/explore_rnn_capacity.py",
            "--d-model", str(d_model),
            "--num-layers", "2",
            "--epochs", "20",
            "--batch-size", "8",
            "--experiment-name", exp_name,
        ]
        success = run_command(cmd, desc)
        results.append((exp_name, success))
    
    return results


def run_depth_experiments():
    """Run depth experiments (varying number of layers)."""
    experiments = [
        (2, "rnn_l2_baseline", "Baseline RNN (2 layers)"),
        (5, "rnn_l5", "Medium Deep RNN (5 layers)"),
        (16, "rnn_l16", "Very Deep RNN (16 layers)"),
    ]
    
    results = []
    for num_layers, exp_name, desc in experiments:
        cmd = [
            sys.executable,
            "experiments/explorations/explore_rnn_capacity.py",
            "--d-model", "256",
            "--num-layers", str(num_layers),
            "--epochs", "20",
            "--batch-size", "8",
            "--experiment-name", exp_name,
        ]
        success = run_command(cmd, desc)
        results.append((exp_name, success))
    
    return results


def run_gru_experiments():
    """Run GRU comparison experiments."""
    experiments = [
        (False, "gru_baseline", "GRU Baseline (unidirectional, 2 layers)"),
    ]
    
    results = []
    for bidirectional, exp_name, desc in experiments:
        cmd = [
            sys.executable,
            "experiments/explorations/run_gru_experiment.py",
            "--d-model", "256",
            "--num-layers", "2",
            "--epochs", "20",
            "--batch-size", "8",
            "--experiment-name", exp_name,
        ]
        if bidirectional:
            cmd.append("--bidirectional")
        
        success = run_command(cmd, desc)
        results.append((exp_name, success))
    
    return results


def print_summary(all_results):
    """Print summary of all experiments."""
    print("\n" + "=" * 80)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 80)
    
    for category, results in all_results.items():
        print(f"\n{category}:")
        for exp_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {status}  {exp_name}")
    
    total_experiments = sum(len(results) for results in all_results.values())
    successful = sum(sum(1 for _, s in results if s) for results in all_results.values())
    
    print(f"\n{'=' * 80}")
    print(f"Total: {successful}/{total_experiments} experiments completed successfully")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Run RNN exploration experiments")
    parser.add_argument(
        "--mode",
        choices=["all", "capacity", "depth", "gru"],
        default="all",
        help="Which experiments to run"
    )
    args = parser.parse_args()
    
    all_results = {}
    
    if args.mode in ["all", "capacity"]:
        print("\n🔬 Running Capacity Experiments (Hidden Dimension)")
        all_results["Capacity Tests"] = run_capacity_experiments()
    
    if args.mode in ["all", "depth"]:
        print("\n🔬 Running Depth Experiments (Number of Layers)")
        all_results["Depth Tests"] = run_depth_experiments()
    
    if args.mode in ["all", "gru"]:
        print("\n🔬 Running GRU Experiments")
        all_results["GRU Comparison"] = run_gru_experiments()
    
    print_summary(all_results)


if __name__ == "__main__":
    main()
