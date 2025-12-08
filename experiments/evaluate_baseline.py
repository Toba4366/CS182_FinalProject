"""
Evaluate the deterministic solver baseline on the ICL dataset.

This script computes the baseline accuracy for each trajectory in the dataset
and saves the results. Can also sweep over demo lengths to find ideal values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch  # type: ignore

from src.datasets.moore_dataset import (
    ICLDatasetConfig,
    load_or_create_icl_samples,
)
from src.fsm.baseline_evaluator import (
    evaluate_deterministic_baseline,
    evaluate_per_trajectory,
)
from src.fsm.trajectory_sampler import TrajectorySamplerConfig


def cleanup_datasets(
    num_states: int | None = None,
    num_actions: int | None = None,
    num_samples: int | None = None,
    data_dir: Path = Path("data"),
) -> int:
    """
    Clean up temporary dataset files created during baseline evaluation.
    
    Args:
        num_states: If specified, only clean datasets with this number of states
        num_actions: If specified, only clean datasets with this number of actions
        num_samples: If specified, only clean datasets with this number of samples
        data_dir: Directory containing the dataset files
    
    Returns:
        Number of files deleted
    """
    if not data_dir.exists():
        return 0
    
    # Pattern for temporary baseline evaluation datasets
    # Format options:
    # - icl_dataset_s{num_states}_a{num_actions}_demo{length}_n{num_samples}.pt
    # - icl_dataset_s{num_states}_a{num_actions}_demo{length}_query{length}_n{num_samples}.pt
    
    # Build base pattern parts (without extension)
    base_parts = ["icl_dataset"]
    if num_states is not None:
        base_parts.append(f"s{num_states}")
    else:
        base_parts.append("s*")
    
    if num_actions is not None:
        base_parts.append(f"a{num_actions}")
    else:
        base_parts.append("a*")
    
    base_parts.append("demo*")
    
    if num_samples is not None:
        base_parts.append(f"n{num_samples}")
    else:
        base_parts.append("n*")
    
    # Create patterns: one without query, one with query
    # Join parts with underscore and add .pt extension
    pattern_without_query = "_".join(base_parts) + ".pt"
    # For pattern with query, insert query* before n*
    pattern_with_query_parts = base_parts.copy()
    n_idx = next(i for i, p in enumerate(pattern_with_query_parts) if p.startswith("n"))
    pattern_with_query_parts.insert(n_idx, "query*")
    pattern_with_query = "_".join(pattern_with_query_parts) + ".pt"
    
    deleted_count = 0
    deleted_files = set()  # Track to avoid double-counting
    
    # Match both patterns
    for pattern in [pattern_without_query, pattern_with_query]:
        for file_path in data_dir.glob(pattern):
            if file_path not in deleted_files:
                try:
                    file_path.unlink()
                    deleted_files.add(file_path)
                    deleted_count += 1
                except OSError as e:
                    print(f"Warning: Could not delete {file_path}: {e}")
    
    return deleted_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate deterministic baseline")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to the dataset file (if None, generates new dataset)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/baseline",
        help="Directory to save results",
    )
    parser.add_argument(
        "--per-trajectory",
        action="store_true",
        help="Also compute per-trajectory accuracies",
    )
    # Configuration parameters
    parser.add_argument(
        "--num-states",
        type=int,
        default=None,
        help="Number of states (overrides dataset config)",
    )
    parser.add_argument(
        "--num-actions",
        type=int,
        default=None,
        help="Number of actions (overrides dataset config)",
    )
    parser.add_argument(
        "--demo-length",
        type=int,
        default=None,
        help="Demo length (overrides dataset config)",
    )
    parser.add_argument(
        "--query-length",
        type=int,
        default=None,
        help="Query length (overrides dataset config)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to generate (if creating new dataset)",
    )
    # Sweep functionality
    parser.add_argument(
        "--sweep-demo-length",
        action="store_true",
        help="Sweep over demo lengths to find ideal value for 99%% accuracy",
    )
    parser.add_argument(
        "--min-demo-length",
        type=int,
        default=10,
        help="Minimum demo length for sweep",
    )
    parser.add_argument(
        "--max-demo-length",
        type=int,
        default=200,
        help="Maximum demo length for sweep",
    )
    parser.add_argument(
        "--demo-length-step",
        type=int,
        default=10,
        help="Step size for demo length sweep",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.995,
        help="Target accuracy to achieve (default: 0.99)",
    )
    parser.add_argument(
        "--stop-early",
        action="store_true",
        help="Stop sweep once target accuracy is reached (continue for a few more steps to verify)",
    )
    parser.add_argument(
        "--verify-steps",
        type=int,
        default=2,
        help="Number of additional steps to verify after reaching target (default: 2)",
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=None,
        help="Configurations to test in format 'states:actions' (e.g., '3:3 5:5 5:8')",
    )
    parser.add_argument(
        "--num-demos",
        type=int,
        default=None,
        help="Number of demos to use per sample (default: all demos, matching model training uses 3)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up temporary dataset files created during evaluation",
    )
    return parser.parse_args()


def evaluate_config(
    num_states: int,
    num_actions: int,
    demo_length: Optional[int],
    query_length: Optional[int],
    num_samples: int,
    seed: int = 42,
    use_cache: bool = True,
    num_demos: Optional[int] = None,
) -> dict:
    """Evaluate a specific configuration."""
    # Create dataset config with unique cache path
    traj_config = TrajectorySamplerConfig(
        num_states=num_states,
        min_actions_per_state=num_actions,
        max_actions_per_state=num_actions,
        seed=seed,
    )
    
    # Create unique cache path based on configuration
    cache_filename = f"icl_dataset_s{num_states}_a{num_actions}"
    if demo_length is not None:
        cache_filename += f"_demo{demo_length}"
    if query_length is not None:
        cache_filename += f"_query{query_length}"
    cache_filename += f"_n{num_samples}.pt"
    
    cache_path = Path("data") / cache_filename
    
    dataset_config = ICLDatasetConfig(
        num_samples=num_samples,
        traj_sampler_config=traj_config,
        demo_length=demo_length,
        query_length=query_length,
        length_variation=0.2,  # Match transformer training (±20% variation)
        cache_path=cache_path,
    )
    
    # Generate samples (will use cache if exists and use_cache=True)
    if use_cache and cache_path.exists():
        print(f"Loading cached dataset: {cache_path}")
        with cache_path.open("rb") as f:
            samples = torch.load(f)
    else:
        print(f"Generating {num_samples} samples with config: {num_states} states, {num_actions} actions, "
              f"demo_length={demo_length}, query_length={query_length}")
        samples = load_or_create_icl_samples(dataset_config)
    
    # Evaluate on all samples
    indices = list(range(len(samples)))
    results = evaluate_deterministic_baseline(samples, indices, num_demos=num_demos)
    
    return results


def sweep_demo_length(
    num_states: int,
    num_actions: int,
    min_demo_length: int,
    max_demo_length: int,
    step: int,
    target_accuracy: float,
    query_length: Optional[int] = None,
    num_samples: int = 1000,
    stop_early: bool = False,
    verify_steps: int = 2,
    num_demos: Optional[int] = None,
) -> dict:
    """Sweep over demo lengths to find the minimum length achieving target accuracy."""
    print(f"\n{'='*70}")
    print(f"Sweeping demo length for {num_states} states, {num_actions} actions")
    print(f"Target accuracy: {target_accuracy:.1%}")
    print(f"Range: {min_demo_length} to {max_demo_length} (step: {step})")
    if stop_early:
        print(f"Early stopping: enabled (verify {verify_steps} steps after target)")
    print(f"{'='*70}\n")
    
    results = []
    ideal_length = None
    steps_since_target = 0
    
    for demo_length in range(min_demo_length, max_demo_length + 1, step):
        result = evaluate_config(
            num_states=num_states,
            num_actions=num_actions,
            demo_length=demo_length,
            query_length=query_length,
            num_samples=num_samples,
            num_demos=num_demos,
        )
        
        accuracy = result["accuracy"]
        results.append({
            "demo_length": demo_length,
            "accuracy": accuracy,
            "num_correct": result["num_correct"],
            "num_samples": result["num_samples"],
        })
        
        status = "✅" if accuracy >= target_accuracy else "❌"
        print(f"  Demo length {demo_length:4d}: {accuracy:.4f} ({status})")
        
        # Check if we've reached target accuracy
        if ideal_length is None and accuracy >= target_accuracy:
            ideal_length = demo_length
            print(f"  🎯 Target accuracy reached at demo_length={demo_length}")
            if stop_early:
                steps_since_target = 1
        elif ideal_length is not None and stop_early:
            steps_since_target += 1
            if steps_since_target > verify_steps:
                print(f"  ✓ Verified target accuracy for {verify_steps} steps, stopping early")
                break
    
    return {
        "config": {
            "num_states": num_states,
            "num_actions": num_actions,
            "query_length": query_length,
        },
        "target_accuracy": target_accuracy,
        "ideal_demo_length": ideal_length,
        "results": results,
    }


def main():
    args = parse_args()
    
    # Parse configurations if provided
    configs = []
    if args.configs:
        for config_str in args.configs:
            try:
                states_str, actions_str = config_str.split(":")
                configs.append((int(states_str), int(actions_str)))
            except ValueError:
                print(f"Warning: Invalid config format '{config_str}', expected 'states:actions'")
                continue
    elif args.sweep_demo_length:
        # Default configurations for sweep
        configs = [(3, 3), (5, 5), (5, 8)]
    elif args.num_states is not None and args.num_actions is not None:
        configs = [(args.num_states, args.num_actions)]
    
    # If sweep mode
    if args.sweep_demo_length:
        all_sweep_results = {}
        
        for num_states, num_actions in configs:
            config_key = f"{num_states}s_{num_actions}a"
            sweep_result = sweep_demo_length(
                num_states=num_states,
                num_actions=num_actions,
                min_demo_length=args.min_demo_length,
                max_demo_length=args.max_demo_length,
                step=args.demo_length_step,
                target_accuracy=args.target_accuracy,
                query_length=args.query_length,
                num_samples=args.num_samples,
                stop_early=args.stop_early,
                verify_steps=args.verify_steps,
                num_demos=args.num_demos,
            )
            all_sweep_results[config_key] = sweep_result
        
        # Save sweep results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / "sweep_results.json"
        with results_path.open("w") as f:
            json.dump(all_sweep_results, f, indent=2)
        
        print(f"\n✅ Sweep results saved to {results_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("SWEEP SUMMARY")
        print("="*70)
        for config_key, sweep_result in all_sweep_results.items():
            ideal = sweep_result["ideal_demo_length"]
            if ideal is not None:
                print(f"{config_key:10s}: Ideal demo_length = {ideal} (accuracy >= {args.target_accuracy:.1%})")
            else:
                print(f"{config_key:10s}: Target accuracy not reached (max tested: {args.max_demo_length})")
        
        # Cleanup if requested
        if args.cleanup:
            print("\n" + "="*70)
            print("CLEANING UP TEMPORARY DATASETS")
            print("="*70)
            total_deleted = 0
            for num_states, num_actions in configs:
                deleted = cleanup_datasets(
                    num_states=num_states,
                    num_actions=num_actions,
                    num_samples=args.num_samples,
                )
                total_deleted += deleted
                if deleted > 0:
                    print(f"  Deleted {deleted} dataset files for {num_states}s_{num_actions}a")
            
            if total_deleted > 0:
                print(f"\n✅ Cleaned up {total_deleted} temporary dataset files")
            else:
                print("\n  No temporary dataset files found to clean up")
        
        return
    
    # Single evaluation mode
    if args.dataset_path:
        # Load existing dataset
        dataset_path = Path(args.dataset_path)
        if not dataset_path.exists():
            print(f"Dataset not found at {dataset_path}")
            return
        print(f"Loading dataset from {dataset_path}")
        with dataset_path.open("rb") as f:
            samples = torch.load(f)
        print(f"Loaded {len(samples)} samples")
    else:
        # Generate new dataset with specified config
        if not configs:
            # Use defaults or provided args
            num_states = args.num_states or 5
            num_actions = args.num_actions or 8
            configs = [(num_states, num_actions)]
        
        num_states, num_actions = configs[0]
        traj_config = TrajectorySamplerConfig(
            num_states=num_states,
            min_actions_per_state=num_actions,
            max_actions_per_state=num_actions,
        )
        
        dataset_config = ICLDatasetConfig(
            num_samples=args.num_samples,
            traj_sampler_config=traj_config,
            demo_length=args.demo_length,
            query_length=args.query_length,
        )
        
        print(f"Generating {args.num_samples} samples...")
        samples = load_or_create_icl_samples(dataset_config)
        print(f"Generated {len(samples)} samples")
    
    # Evaluate on all samples (or split if dataset is large enough)
    if len(samples) >= 10_000:
        train_indices = list(range(0, 6_000))
        val_indices = list(range(6_000, 8_000))
        test_indices = list(range(8_000, 10_000))
        splits = [("train", train_indices), ("val", val_indices), ("test", test_indices)]
    else:
        # Use all samples
        splits = [("all", list(range(len(samples))))]
    
    results = {}
    
    for split_name, indices in splits:
        if len(indices) > len(samples):
            print(f"Warning: {split_name} indices exceed dataset size, using available samples")
            indices = [i for i in indices if i < len(samples)]
        
        print(f"\nEvaluating {split_name} set ({len(indices)} samples)...")
        
        # Overall accuracy
        split_results = evaluate_deterministic_baseline(samples, indices, num_demos=args.num_demos)
        results[split_name] = split_results
        
        print(f"  Accuracy: {split_results['accuracy']:.4f}")
        print(f"  Correct (100%): {split_results['num_correct']}/{split_results['num_samples']}")
        print(f"  Incorrect: {split_results['num_incorrect']}/{split_results['num_samples']}")
        
        # Per-trajectory accuracy if requested
        if args.per_trajectory:
            per_traj = evaluate_per_trajectory(samples, indices, num_demos=args.num_demos)
            results[f"{split_name}_per_trajectory"] = per_traj
            print(f"  Per-trajectory results saved ({len(per_traj)} trajectories)")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "baseline_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {results_path}")
    
    # Cleanup if requested (for single evaluation mode)
    if args.cleanup:
        print("\n" + "="*60)
        print("CLEANING UP TEMPORARY DATASETS")
        print("="*60)
        if configs:
            num_states, num_actions = configs[0]
            deleted = cleanup_datasets(
                num_states=num_states,
                num_actions=num_actions,
                num_samples=args.num_samples,
            )
            if deleted > 0:
                print(f"✅ Cleaned up {deleted} temporary dataset files")
            else:
                print("  No temporary dataset files found to clean up")
        else:
            # Clean all temporary baseline datasets
            deleted = cleanup_datasets()
            if deleted > 0:
                print(f"✅ Cleaned up {deleted} temporary dataset files")
            else:
                print("  No temporary dataset files found to clean up")
    
    # Print summary
    print("\n" + "="*60)
    print("BASELINE ACCURACY SUMMARY")
    print("="*60)
    for split_name in results.keys():
        if not split_name.endswith("_per_trajectory"):
            acc = results[split_name]["accuracy"]
            print(f"{split_name.upper():8s}: {acc:.4f} ({results[split_name]['num_correct']}/{results[split_name]['num_samples']} perfect)")


if __name__ == "__main__":
    main()

