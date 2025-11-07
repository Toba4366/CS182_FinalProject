"""
Data utilities for saving and loading experiment data.
"""

import json
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime


def save_experiment_data(data: Dict[str, Any], 
                        filepath: str,
                        format: str = 'json') -> None:
    """
    Save experiment data to file.
    
    Args:
        data: Dictionary containing experiment data
        filepath: Path to save file
        format: Save format ('json', 'pickle', 'csv')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        # Convert numpy arrays and tensors to lists for JSON serialization
        serializable_data = _make_json_serializable(data)
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    elif format == 'csv':
        # Convert data to DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_experiment_data(filepath: str,
                        format: str = 'auto') -> Dict[str, Any]:
    """
    Load experiment data from file.
    
    Args:
        filepath: Path to data file
        format: Load format ('json', 'pickle', 'csv', 'auto')
        
    Returns:
        Dictionary containing loaded data
    """
    filepath = Path(filepath)
    
    if format == 'auto':
        # Infer format from file extension
        ext = filepath.suffix.lower()
        if ext == '.json':
            format = 'json'
        elif ext in ['.pkl', '.pickle']:
            format = 'pickle'
        elif ext == '.csv':
            format = 'csv'
        else:
            raise ValueError(f"Cannot infer format from extension: {ext}")
    
    if format == 'json':
        with open(filepath, 'r') as f:
            return json.load(f)
    
    elif format == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    elif format == 'csv':
        df = pd.read_csv(filepath)
        return df.to_dict('list')
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def _make_json_serializable(obj):
    """Convert object to JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # Try to convert to string as fallback
        return str(obj)


def create_experiment_summary(results: Dict[str, Any],
                            config: Dict[str, Any],
                            save_path: str) -> None:
    """
    Create a summary report of experiment results.
    
    Args:
        results: Experiment results dictionary
        config: Experiment configuration dictionary
        save_path: Path to save summary
    """
    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiment_config': config,
        'results_summary': {},
        'model_performance': {},
        'training_info': {}
    }
    
    # Extract key metrics
    if 'train_losses' in results:
        train_losses = results['train_losses']
        summary['training_info'].update({
            'final_train_loss': train_losses[-1] if train_losses else None,
            'min_train_loss': min(train_losses) if train_losses else None,
            'training_steps': len(train_losses)
        })
    
    if 'val_losses' in results:
        val_losses = results['val_losses']
        summary['training_info'].update({
            'final_val_loss': val_losses[-1] if val_losses else None,
            'min_val_loss': min(val_losses) if val_losses else None,
            'best_val_loss': results.get('best_val_loss', None)
        })
    
    # Add evaluation metrics if available
    for key in ['accuracy', 'exact_match_rate', 'perplexity']:
        if key in results:
            summary['model_performance'][key] = results[key]
    
    # Save summary
    save_experiment_data(summary, save_path, format='json')


def aggregate_multiple_runs(experiment_dirs: List[str],
                           output_path: str) -> Dict[str, Any]:
    """
    Aggregate results from multiple experimental runs.
    
    Args:
        experiment_dirs: List of paths to experiment directories
        output_path: Path to save aggregated results
        
    Returns:
        Aggregated results dictionary
    """
    all_results = []
    all_configs = []
    
    for exp_dir in experiment_dirs:
        exp_path = Path(exp_dir)
        
        # Load results
        results_file = exp_path / 'training_history.json'
        if results_file.exists():
            results = load_experiment_data(str(results_file))
            all_results.append(results)
        
        # Load config
        config_file = exp_path / 'config.yaml'
        if config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            all_configs.append(config)
    
    if not all_results:
        raise ValueError("No valid experiment results found")
    
    # Aggregate metrics
    aggregated = {
        'num_runs': len(all_results),
        'configs': all_configs,
        'aggregated_metrics': {}
    }
    
    # Compute statistics for final metrics
    metric_names = ['best_val_loss']
    for metric in metric_names:
        values = [r.get(metric) for r in all_results if metric in r]
        if values:
            aggregated['aggregated_metrics'][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
    
    # Save aggregated results
    save_experiment_data(aggregated, output_path, format='json')
    
    return aggregated


def export_results_for_analysis(experiment_dir: str,
                               output_format: str = 'csv') -> None:
    """
    Export experiment results in format suitable for external analysis.
    
    Args:
        experiment_dir: Path to experiment directory
        output_format: Export format ('csv', 'excel')
    """
    exp_path = Path(experiment_dir)
    
    # Load training history
    results_file = exp_path / 'training_history.json'
    if not results_file.exists():
        raise FileNotFoundError(f"No training history found in {experiment_dir}")
    
    results = load_experiment_data(str(results_file))
    
    # Create DataFrames
    dfs = {}
    
    # Training curves
    if 'train_losses' in results:
        dfs['training_curves'] = pd.DataFrame({
            'step': range(len(results['train_losses'])),
            'train_loss': results['train_losses']
        })
        
        if 'val_losses' in results:
            # Interpolate validation losses to match training steps
            val_losses = results['val_losses']
            val_steps = np.linspace(0, len(results['train_losses']) - 1, len(val_losses))
            
            val_df = pd.DataFrame({
                'step': val_steps.astype(int),
                'val_loss': val_losses
            })
            
            dfs['training_curves'] = dfs['training_curves'].merge(
                val_df, on='step', how='left'
            )
    
    # Export DataFrames
    for name, df in dfs.items():
        if output_format == 'csv':
            output_file = exp_path / f'{name}.csv'
            df.to_csv(output_file, index=False)
        elif output_format == 'excel':
            output_file = exp_path / f'{name}.xlsx'
            df.to_excel(output_file, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")


def compare_experiments(experiment_dirs: List[str],
                       metrics: List[str] = ['best_val_loss'],
                       save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Compare metrics across multiple experiments.
    
    Args:
        experiment_dirs: List of experiment directory paths
        metrics: List of metrics to compare
        save_path: Optional path to save comparison table
        
    Returns:
        Comparison DataFrame
    """
    comparison_data = []
    
    for exp_dir in experiment_dirs:
        exp_path = Path(exp_dir)
        exp_name = exp_path.name
        
        # Load results
        results_file = exp_path / 'training_history.json'
        if results_file.exists():
            results = load_experiment_data(str(results_file))
            
            row = {'experiment': exp_name}
            for metric in metrics:
                row[metric] = results.get(metric, None)
            
            comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    if save_path:
        df.to_csv(save_path, index=False)
    
    return df