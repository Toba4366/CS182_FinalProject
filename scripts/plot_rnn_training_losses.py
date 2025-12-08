#!/usr/bin/env python3
"""
Script to visualize training losses for rnn_3s3a and rnn_5s5a experiments.

Creates comparison plots showing training loss curves for both configurations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import plotly.graph_objects as go


def load_summary(summary_path: Path) -> Dict:
    """Load the summary.json file."""
    with open(summary_path, "r") as f:
        return json.load(f)


def extract_training_losses(summary: Dict, stage_idx: int = 0) -> Tuple[List[int], List[List[float]]]:
    """
    Extract training losses for a specific stage per run.
    
    Returns:
        (epochs, train_losses_per_run)
    """
    runs = summary["runs"]
    
    # Collect all training losses for this stage across runs
    all_train_losses = []
    max_epochs = 0
    
    for run in runs:
        stage_metrics = run["stage_metrics"]
        if stage_idx < len(stage_metrics) and stage_metrics[stage_idx]:
            metrics = stage_metrics[stage_idx]
            epochs = [m["epoch"] for m in metrics]
            train_losses = [m["train_loss"] for m in metrics]
            
            all_train_losses.append(train_losses)
            max_epochs = max(max_epochs, len(epochs))
    
    if not all_train_losses:
        return [], []
    
    # Align all runs to the same epoch length (pad with last value if needed)
    epochs = list(range(1, max_epochs + 1))
    
    # Pad shorter runs with their last value
    padded_train_losses = []
    
    for train_losses in all_train_losses:
        last_train_loss = train_losses[-1] if train_losses else 0.0
        padded_train_losses.append(
            train_losses + [last_train_loss] * (max_epochs - len(train_losses))
        )
    
    return epochs, padded_train_losses


def plot_comparison(
    epochs_3s3a: List[int],
    train_losses_3s3a: List[List[float]],
    epochs_5s5a: List[int],
    train_losses_5s5a: List[List[float]],
    output_path: Path,
):
    """Plot training loss comparison for 3s3a and 5s5a."""
    # Calculate mean and std for shaded regions
    train_losses_3s3a_mean = np.mean(train_losses_3s3a, axis=0)
    train_losses_3s3a_std = np.std(train_losses_3s3a, axis=0)
    train_losses_5s5a_mean = np.mean(train_losses_5s5a, axis=0)
    train_losses_5s5a_std = np.std(train_losses_5s5a, axis=0)
    
    # Color palette
    color_3s3a = "#1f77b4"  # Blue
    color_5s5a = "#ff7f0e"  # Orange
    
    fig = go.Figure()
    
    # Add shaded region for 3s3a (mean ± std)
    fig.add_trace(go.Scatter(
        x=list(epochs_3s3a) + list(epochs_3s3a[::-1]),
        y=list(train_losses_3s3a_mean + train_losses_3s3a_std) + 
          list((train_losses_3s3a_mean - train_losses_3s3a_std)[::-1]),
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
        name="3s3a std",
    ))
    
    # Add mean line for 3s3a
    fig.add_trace(go.Scatter(
        x=epochs_3s3a,
        y=train_losses_3s3a_mean,
        mode="lines",
        name="3s3a (Mean)",
        line=dict(color=color_3s3a, width=2),
        legendgroup="3s3a",
    ))
    
    # Add individual run lines for 3s3a
    for run_id, train_losses in enumerate(train_losses_3s3a):
        fig.add_trace(go.Scatter(
            x=epochs_3s3a,
            y=train_losses,
            mode="lines",
            name=f"3s3a Run {run_id + 1}",
            line=dict(color=color_3s3a, width=1.5),
            opacity=0.4,
            legendgroup="3s3a",
            showlegend=False,
        ))
    
    # Add shaded region for 5s5a (mean ± std)
    fig.add_trace(go.Scatter(
        x=list(epochs_5s5a) + list(epochs_5s5a[::-1]),
        y=list(train_losses_5s5a_mean + train_losses_5s5a_std) + 
          list((train_losses_5s5a_mean - train_losses_5s5a_std)[::-1]),
        fill="toself",
        fillcolor="rgba(255, 127, 14, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
        name="5s5a std",
    ))
    
    # Add mean line for 5s5a
    fig.add_trace(go.Scatter(
        x=epochs_5s5a,
        y=train_losses_5s5a_mean,
        mode="lines",
        name="5s5a (Mean)",
        line=dict(color=color_5s5a, width=2),
        legendgroup="5s5a",
    ))
    
    # Add individual run lines for 5s5a
    for run_id, train_losses in enumerate(train_losses_5s5a):
        fig.add_trace(go.Scatter(
            x=epochs_5s5a,
            y=train_losses,
            mode="lines",
            name=f"5s5a Run {run_id + 1}",
            line=dict(color=color_5s5a, width=1.5),
            opacity=0.4,
            legendgroup="5s5a",
            showlegend=False,
        ))
    
    fig.update_layout(
        title=dict(
            text="Training Loss Comparison: RNN 3s3a vs 5s5a<br><sub>RNN Architecture: 2 layers, 128 dim (~68K parameters)</sub>",
            font=dict(size=22, family="Arial Black"),
        ),
        xaxis_title="Epoch",
        yaxis_title="Training Loss",
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=16),
        ),
        yaxis=dict(
            range=[0, None],  # Start y-axis at 0
            title_font=dict(size=18),
            tickfont=dict(size=16),
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            font=dict(size=16),
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
        width=1200,
        height=700,
    )
    
    # Save as HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    print(f"✓ Saved comparison plot to: {output_path}")
    
    # Also save as PNG if possible
    try:
        png_path = output_path.with_suffix(".png")
        fig.write_image(str(png_path), width=1200, height=700, scale=2)
        print(f"✓ Saved PNG to: {png_path}")
    except Exception as e:
        print(f"  (PNG export failed: {e})")


def main():
    parser = argparse.ArgumentParser(description="Plot training losses for RNN 3s3a and 5s5a")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/rnn_ablation",
        help="Directory containing rnn_3s3a and rnn_5s5a subdirectories",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/rnn_ablation/training_loss_comparison.html",
        help="Output path for the plot",
    )
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    
    # Load summaries
    summary_3s3a_path = results_dir / "rnn_3s3a" / "summary.json"
    summary_5s5a_path = results_dir / "rnn_5s5a" / "summary.json"
    
    if not summary_3s3a_path.exists():
        print(f"Error: {summary_3s3a_path} not found")
        return
    
    if not summary_5s5a_path.exists():
        print(f"Error: {summary_5s5a_path} not found")
        return
    
    print(f"Loading {summary_3s3a_path}...")
    summary_3s3a = load_summary(summary_3s3a_path)
    
    print(f"Loading {summary_5s5a_path}...")
    summary_5s5a = load_summary(summary_5s5a_path)
    
    # Extract training losses
    print("Extracting training losses...")
    epochs_3s3a, train_losses_3s3a = extract_training_losses(summary_3s3a, stage_idx=0)
    epochs_5s5a, train_losses_5s5a = extract_training_losses(summary_5s5a, stage_idx=0)
    
    if not train_losses_3s3a:
        print("Error: No training losses found for 3s3a")
        return
    
    if not train_losses_5s5a:
        print("Error: No training losses found for 5s5a")
        return
    
    print(f"3s3a: {len(train_losses_3s3a)} runs, {len(epochs_3s3a)} epochs")
    print(f"5s5a: {len(train_losses_5s5a)} runs, {len(epochs_5s5a)} epochs")
    
    # Plot comparison
    print("Generating plot...")
    plot_comparison(
        epochs_3s3a,
        train_losses_3s3a,
        epochs_5s5a,
        train_losses_5s5a,
        output_path,
    )
    
    print("Done!")


if __name__ == "__main__":
    main()

