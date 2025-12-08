"""
Script to visualize ablation study results from summary.json.

Creates graphs for curriculum stages and query length extrapolation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_summary(summary_path: Path) -> Dict:
    """Load the summary.json file."""
    with open(summary_path, "r") as f:
        return json.load(f)


def extract_stage_metrics_per_run(
    summary: Dict, stage_idx: int
) -> Tuple[List[int], List[List[float]], List[List[float]], List[List[float]], List[List[float]]]:
    """
    Extract metrics for a specific stage per run (for line graphs).
    
    Returns:
        (epochs, train_losses_per_run, val_losses_per_run, train_accs_per_run, val_accs_per_run)
    """
    runs = summary["runs"]
    
    # Collect all metrics for this stage across runs
    all_train_losses = []
    all_val_losses = []
    all_train_accs = []
    all_val_accs = []
    max_epochs = 0
    
    for run in runs:
        stage_metrics = run["stage_metrics"]
        if stage_idx < len(stage_metrics) and stage_metrics[stage_idx]:
            metrics = stage_metrics[stage_idx]
            epochs = [m["epoch"] for m in metrics]
            train_losses = [m["train_loss"] for m in metrics]
            val_losses = [m["val_loss"] for m in metrics]
            train_accs = [m["train_acc"] for m in metrics]
            val_accs = [m["val_acc"] for m in metrics]
            
            all_train_losses.append(train_losses)
            all_val_losses.append(val_losses)
            all_train_accs.append(train_accs)
            all_val_accs.append(val_accs)
            max_epochs = max(max_epochs, len(epochs))
    
    if not all_train_losses:
        return [], [], [], [], []
    
    # Align all runs to the same epoch length (pad with last value if needed)
    epochs = list(range(1, max_epochs + 1))
    
    # Pad shorter runs with their last value
    padded_train_losses = []
    padded_val_losses = []
    padded_train_accs = []
    padded_val_accs = []
    
    for train_losses, val_losses, train_accs, val_accs in zip(
        all_train_losses, all_val_losses, all_train_accs, all_val_accs
    ):
        last_train_loss = train_losses[-1] if train_losses else 0.0
        last_val_loss = val_losses[-1] if val_losses else 0.0
        last_train_acc = train_accs[-1] if train_accs else 0.0
        last_val_acc = val_accs[-1] if val_accs else 0.0
        
        padded_train_losses.append(
            train_losses + [last_train_loss] * (max_epochs - len(train_losses))
        )
        padded_val_losses.append(
            val_losses + [last_val_loss] * (max_epochs - len(val_losses))
        )
        padded_train_accs.append(
            train_accs + [last_train_acc] * (max_epochs - len(train_accs))
        )
        padded_val_accs.append(
            val_accs + [last_val_acc] * (max_epochs - len(val_accs))
        )
    
    return (
        epochs,
        padded_train_losses,
        padded_val_losses,
        padded_train_accs,
        padded_val_accs,
    )


def plot_stage_accuracy(
    epochs: List[int],
    train_accs_per_run: List[List[float]],
    val_accs_per_run: List[List[float]],
    stage_num: int,
    output_dir: Path,
    num_states: int | None = None,
    num_actions: int | None = None,
    total_stages: int = 2,
    architecture_info: str | None = None,
):
    """Plot accuracy curves for a curriculum stage with separate plots for train and validation."""
    # Calculate mean and std for shaded regions
    train_accs_mean = np.mean(train_accs_per_run, axis=0)
    train_accs_std = np.std(train_accs_per_run, axis=0)
    val_accs_mean = np.mean(val_accs_per_run, axis=0)
    val_accs_std = np.std(val_accs_per_run, axis=0)
    
    # Color palette for runs
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    
    # Plot training accuracy
    fig = go.Figure()
    
    # Add shaded region for mean ± std
    fig.add_trace(go.Scatter(
        x=epochs + epochs[::-1],
        y=list(train_accs_mean + train_accs_std) + list((train_accs_mean - train_accs_std)[::-1]),
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
    ))
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_accs_mean,
        mode="lines",
        name="Mean",
        line=dict(color="steelblue", width=4),
    ))
    
    # Add individual run lines
    for run_id, train_accs in enumerate(train_accs_per_run):
        fig.add_trace(go.Scatter(
            x=epochs,
            y=train_accs,
            mode="lines",
            name=f"Run {run_id + 1}",
            line=dict(color=colors[run_id % len(colors)], width=1.5),
            opacity=0.6,
        ))
    
    # Build title with stage configuration
    title_suffix = ""
    if num_states is not None and num_actions is not None:
        title_suffix = f" ({num_states} states, {num_actions} actions)"
    
    # Add architecture info as subtitle if provided
    arch_subtitle = ""
    if architecture_info:
        arch_subtitle = f"<br><sub>{architecture_info}</sub>"
    
    # Only include "Curriculum Stage X" if there are multiple stages
    stage_prefix = f"Curriculum Stage {stage_num} - " if total_stages > 1 else ""
    
    fig.update_layout(
        title=dict(
            text=f"{stage_prefix}Training Accuracy{title_suffix}{arch_subtitle}",
            font=dict(size=22, family="Arial Black"),
        ),
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=16),
        ),
        yaxis=dict(
            range=[0, 1.05],
            title_font=dict(size=18),
            tickfont=dict(size=16),
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(font=dict(size=16)),
        width=1000,
        height=600,
    )
    
    # Add transformer prefix if architecture info indicates transformer
    prefix = "transformer_" if architecture_info and "Transformer" in architecture_info else ""
    train_path = output_dir / f"{prefix}stage{stage_num}_train_accuracy.html"
    fig.write_html(str(train_path))
    # Also save as PNG image
    try:
        png_path = train_path.with_suffix(".png")
        fig.write_image(str(png_path), width=1000, height=600, scale=2)
        print(f"✓ Saved training accuracy plot to: {train_path} and {png_path}")
    except Exception as e:
        print(f"✓ Saved training accuracy plot to: {train_path} (PNG export failed: {e})")
    
    # Plot validation accuracy
    fig = go.Figure()
    
    # Add shaded region for mean ± std
    fig.add_trace(go.Scatter(
        x=epochs + epochs[::-1],
        y=list(val_accs_mean + val_accs_std) + list((val_accs_mean - val_accs_std)[::-1]),
        fill="toself",
        fillcolor="rgba(214, 39, 40, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
    ))
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_accs_mean,
        mode="lines",
        name="Mean",
        line=dict(color="crimson", width=4),
    ))
    
    # Add individual run lines
    for run_id, val_accs in enumerate(val_accs_per_run):
        fig.add_trace(go.Scatter(
            x=epochs,
            y=val_accs,
            mode="lines",
            name=f"Run {run_id + 1}",
            line=dict(color=colors[run_id % len(colors)], width=1.5),
            opacity=0.6,
        ))
    
    # Build title with stage configuration
    title_suffix = ""
    if num_states is not None and num_actions is not None:
        title_suffix = f" ({num_states} states, {num_actions} actions)"
    
    # Add architecture info as subtitle if provided
    arch_subtitle = ""
    if architecture_info:
        arch_subtitle = f"<br><sub>{architecture_info}</sub>"
    
    # Only include "Curriculum Stage X" if there are multiple stages
    stage_prefix = f"Curriculum Stage {stage_num} - " if total_stages > 1 else ""
    
    fig.update_layout(
        title=dict(
            text=f"{stage_prefix}Validation Accuracy{title_suffix}{arch_subtitle}",
            font=dict(size=22, family="Arial Black"),
        ),
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=16),
        ),
        yaxis=dict(
            range=[0, 1.05],
            title_font=dict(size=18),
            tickfont=dict(size=16),
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(font=dict(size=16)),
        width=1000,
        height=600,
    )
    
    # Add transformer prefix if architecture info indicates transformer
    prefix = "transformer_" if architecture_info and "Transformer" in architecture_info else ""
    val_path = output_dir / f"{prefix}stage{stage_num}_val_accuracy.html"
    fig.write_html(str(val_path))
    # Also save as PNG image
    try:
        png_path = val_path.with_suffix(".png")
        fig.write_image(str(png_path), width=1000, height=600, scale=2)
        print(f"✓ Saved validation accuracy plot to: {val_path} and {png_path}")
    except Exception as e:
        print(f"✓ Saved validation accuracy plot to: {val_path} (PNG export failed: {e})")


def plot_stage_loss(
    epochs: List[int],
    train_losses_per_run: List[List[float]],
    val_losses_per_run: List[List[float]],
    stage_num: int,
    output_dir: Path,
    num_states: int | None = None,
    num_actions: int | None = None,
    total_stages: int = 2,
    architecture_info: str | None = None,
):
    """Plot loss curves for a curriculum stage with separate plots for train and validation."""
    # Calculate mean and std for shaded regions
    train_losses_mean = np.mean(train_losses_per_run, axis=0)
    train_losses_std = np.std(train_losses_per_run, axis=0)
    val_losses_mean = np.mean(val_losses_per_run, axis=0)
    val_losses_std = np.std(val_losses_per_run, axis=0)
    
    # Color palette for runs
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    
    # Plot training loss
    fig = go.Figure()
    
    # Add shaded region for mean ± std
    fig.add_trace(go.Scatter(
        x=epochs + epochs[::-1],
        y=list(train_losses_mean + train_losses_std) + list((train_losses_mean - train_losses_std)[::-1]),
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
    ))
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=epochs,
        y=train_losses_mean,
        mode="lines",
        name="Mean",
        line=dict(color="steelblue", width=4),
    ))
    
    # Add individual run lines
    for run_id, train_losses in enumerate(train_losses_per_run):
        fig.add_trace(go.Scatter(
            x=epochs,
            y=train_losses,
            mode="lines",
            name=f"Run {run_id + 1}",
            line=dict(color=colors[run_id % len(colors)], width=1.5),
            opacity=0.6,
        ))
    
    # Build title with stage configuration
    title_suffix = ""
    if num_states is not None and num_actions is not None:
        title_suffix = f" ({num_states} states, {num_actions} actions)"
    
    # Add architecture info as subtitle if provided
    arch_subtitle = ""
    if architecture_info:
        arch_subtitle = f"<br><sub>{architecture_info}</sub>"
    
    # Only include "Curriculum Stage X" if there are multiple stages
    stage_prefix = f"Curriculum Stage {stage_num} - " if total_stages > 1 else ""
    
    fig.update_layout(
        title=dict(
            text=f"{stage_prefix}Training Loss{title_suffix}{arch_subtitle}",
            font=dict(size=22, family="Arial Black"),
        ),
        xaxis_title="Epoch",
        yaxis_title="Loss",
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=16),
        ),
        yaxis=dict(
            range=[0, None],
            title_font=dict(size=18),
            tickfont=dict(size=16),
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(font=dict(size=16)),
        width=1000,
        height=600,
    )
    
    # Add transformer prefix if architecture info indicates transformer
    prefix = "transformer_" if architecture_info and "Transformer" in architecture_info else ""
    train_path = output_dir / f"{prefix}stage{stage_num}_train_loss.html"
    fig.write_html(str(train_path))
    # Also save as PNG image
    try:
        png_path = train_path.with_suffix(".png")
        fig.write_image(str(png_path), width=1000, height=600, scale=2)
        print(f"✓ Saved training loss plot to: {train_path} and {png_path}")
    except Exception as e:
        print(f"✓ Saved training loss plot to: {train_path} (PNG export failed: {e})")
    
    # Plot validation loss
    fig = go.Figure()
    
    # Add shaded region for mean ± std
    fig.add_trace(go.Scatter(
        x=epochs + epochs[::-1],
        y=list(val_losses_mean + val_losses_std) + list((val_losses_mean - val_losses_std)[::-1]),
        fill="toself",
        fillcolor="rgba(214, 39, 40, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
    ))
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=epochs,
        y=val_losses_mean,
        mode="lines",
        name="Mean",
        line=dict(color="crimson", width=4),
    ))
    
    # Add individual run lines
    for run_id, val_losses in enumerate(val_losses_per_run):
        fig.add_trace(go.Scatter(
            x=epochs,
            y=val_losses,
            mode="lines",
            name=f"Run {run_id + 1}",
            line=dict(color=colors[run_id % len(colors)], width=1.5),
            opacity=0.6,
        ))
    
    # Build title with stage configuration
    title_suffix = ""
    if num_states is not None and num_actions is not None:
        title_suffix = f" ({num_states} states, {num_actions} actions)"
    
    # Add architecture info as subtitle if provided
    arch_subtitle = ""
    if architecture_info:
        arch_subtitle = f"<br><sub>{architecture_info}</sub>"
    
    # Only include "Curriculum Stage X" if there are multiple stages
    stage_prefix = f"Curriculum Stage {stage_num} - " if total_stages > 1 else ""
    
    fig.update_layout(
        title=dict(
            text=f"{stage_prefix}Validation Loss{title_suffix}{arch_subtitle}",
            font=dict(size=22, family="Arial Black"),
        ),
        xaxis_title="Epoch",
        yaxis_title="Loss",
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=16),
        ),
        yaxis=dict(
            range=[0, None],
            title_font=dict(size=18),
            tickfont=dict(size=16),
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(font=dict(size=16)),
        width=1000,
        height=600,
    )
    
    # Add transformer prefix if architecture info indicates transformer
    prefix = "transformer_" if architecture_info and "Transformer" in architecture_info else ""
    val_path = output_dir / f"{prefix}stage{stage_num}_val_loss.html"
    fig.write_html(str(val_path))
    # Also save as PNG image
    try:
        png_path = val_path.with_suffix(".png")
        fig.write_image(str(png_path), width=1000, height=600, scale=2)
        print(f"✓ Saved validation loss plot to: {val_path} and {png_path}")
    except Exception as e:
        print(f"✓ Saved validation loss plot to: {val_path} (PNG export failed: {e})")


def extract_query_length_results(summary: Dict) -> Dict:
    """Extract query length test results across all runs."""
    runs = summary["runs"]
    
    # Collect results for each query length
    query_lengths = []
    test_accs = {}
    baseline_accs = {}
    
    for run in runs:
        query_results = run.get("query_length_results", {})
        for query_len, results in query_results.items():
            query_len = int(query_len)
            if query_len not in query_lengths:
                query_lengths.append(query_len)
                test_accs[query_len] = []
                baseline_accs[query_len] = []
            
            test_accs[query_len].append(results.get("test_accuracy", 0.0))
            if "baseline_accuracy" in results:
                baseline_accs[query_len].append(results["baseline_accuracy"])
    
    # Also get initial test accuracy
    initial_test_accs = []
    for run in runs:
        if "initial_test_accuracy" in run:
            initial_test_accs.append(run["initial_test_accuracy"])
    
    return {
        "query_lengths": sorted(query_lengths),
        "test_accs": test_accs,
        "baseline_accs": baseline_accs,
        "initial_test_accs": initial_test_accs,
    }


def create_query_length_table(
    query_lengths: List[int],
    test_means: List[float],
    test_stds: List[float],
    baseline_means: List[float | None],
    output_path: Path,
):
    """Create a table with query length extrapolation results."""
    # Prepare table data (convert to percentages)
    table_data = []
    for qlen, mean, std, baseline in zip(query_lengths, test_means, test_stds, baseline_means):
        row = {
            "Query Length": str(qlen),
            "Model Test Accuracy (Mean)": f"{mean * 100:.3f}%",
            "Model Test Accuracy (Std)": f"{std * 100:.3f}%",
            "Baseline Accuracy": f"{baseline * 100:.3f}%" if baseline is not None else "N/A",
        }
        table_data.append(row)
    
    # Create Plotly table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(table_data[0].keys()),
            fill_color="paleturquoise",
            align="left",
            font=dict(size=16, color="black"),
        ),
        cells=dict(
            values=[[row[k] for row in table_data] for k in table_data[0].keys()],
            fill_color="white",
            align="left",
            font=dict(size=14),
        ),
    )])
    
    fig.update_layout(
        title=dict(
            text="Query Length Extrapolation Results",
            font=dict(size=22, family="Arial Black"),
        ),
        width=800,
        height=300,
    )
    
    # Save as HTML
    table_path = output_path.parent / f"{output_path.stem}_table.html"
    fig.write_html(str(table_path))
    print(f"✓ Saved query length extrapolation table to: {table_path}")


def plot_combined_stages(
    summary: Dict,
    output_dir: Path,
    architecture_info: str | None = None,
):
    """Create a combined plot showing all metrics for both stages (or single stage)."""
    # Extract metrics for both stages
    (
        epochs1,
        train_losses_per_run1,
        val_losses_per_run1,
        train_accs_per_run1,
        val_accs_per_run1,
    ) = extract_stage_metrics_per_run(summary, stage_idx=0)
    
    (
        epochs2,
        train_losses_per_run2,
        val_losses_per_run2,
        train_accs_per_run2,
        val_accs_per_run2,
    ) = extract_stage_metrics_per_run(summary, stage_idx=1)
    
    if not epochs1 and not epochs2:
        print("⚠ No data for combined plot")
        return
    
    # Get stage configurations
    stage1_config = summary.get("curriculum_stages", [{}])[0] if summary.get("curriculum_stages") else {}
    stage2_config = summary.get("curriculum_stages", [{}])[1] if len(summary.get("curriculum_stages", [])) > 1 else {}
    num_states1 = stage1_config.get("num_states")
    num_actions1 = stage1_config.get("max_actions_per_state")
    num_states2 = stage2_config.get("num_states")
    num_actions2 = stage2_config.get("max_actions_per_state")
    
    # Determine if single stage or two stages
    has_stage2 = bool(epochs2)
    
    # Create subplots based on number of stages
    if has_stage2:
        # Two stages: 4 rows x 2 cols
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                f"Stage 1 Training Loss ({num_states1}s{num_actions1}a)" if num_states1 and num_actions1 else "Stage 1 Training Loss",
                f"Stage 1 Training Accuracy ({num_states1}s{num_actions1}a)" if num_states1 and num_actions1 else "Stage 1 Training Accuracy",
                f"Stage 1 Validation Loss ({num_states1}s{num_actions1}a)" if num_states1 and num_actions1 else "Stage 1 Validation Loss",
                f"Stage 1 Validation Accuracy ({num_states1}s{num_actions1}a)" if num_states1 and num_actions1 else "Stage 1 Validation Accuracy",
                f"Stage 2 Training Loss ({num_states2}s{num_actions2}a)" if num_states2 and num_actions2 else "Stage 2 Training Loss",
                f"Stage 2 Training Accuracy ({num_states2}s{num_actions2}a)" if num_states2 and num_actions2 else "Stage 2 Training Accuracy",
                f"Stage 2 Validation Loss ({num_states2}s{num_actions2}a)" if num_states2 and num_actions2 else "Stage 2 Validation Loss",
                f"Stage 2 Validation Accuracy ({num_states2}s{num_actions2}a)" if num_states2 and num_actions2 else "Stage 2 Validation Accuracy",
            ),
            vertical_spacing=0.05,
            horizontal_spacing=0.06,
        )
        num_rows = 4
        num_cols = 2
    else:
        # Single stage: 2 rows x 2 cols (square layout)
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"Training Loss ({num_states1}s{num_actions1}a)" if num_states1 and num_actions1 else "Training Loss",
                f"Training Accuracy ({num_states1}s{num_actions1}a)" if num_states1 and num_actions1 else "Training Accuracy",
                f"Validation Loss ({num_states1}s{num_actions1}a)" if num_states1 and num_actions1 else "Validation Loss",
                f"Validation Accuracy ({num_states1}s{num_actions1}a)" if num_states1 and num_actions1 else "Validation Accuracy",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
        )
        num_rows = 2
        num_cols = 2
    
    # Update subplot titles to be more visible
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 16
        fig.layout.annotations[i].font.family = "Arial Black"
    
    # Color palette
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    
    # Helper function to add a plot to a subplot
    def add_plot_to_subplot(row, col, epochs, data_per_run, data_type="loss", stage_num=1):
        if not epochs or not data_per_run:
            return
        
        data_mean = np.mean(data_per_run, axis=0)
        data_std = np.std(data_per_run, axis=0)
        
        # Add shaded region
        fig.add_trace(
            go.Scatter(
                x=list(epochs) + list(epochs[::-1]),
                y=list(data_mean + data_std) + list((data_mean - data_std)[::-1]),
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.2)" if data_type == "loss" else "rgba(31, 119, 180, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row, col=col
        )
        
        # Add mean line
        line_color = "steelblue" if data_type == "loss" else "steelblue"
        mean_name = "Mean (Training)"
        if data_type == "accuracy":
            line_color = "steelblue"
            mean_name = "Mean (Training)"
        elif data_type == "val_loss":
            line_color = "crimson"
            mean_name = "Mean (Validation)"
        elif data_type == "val_acc":
            line_color = "crimson"
            mean_name = "Mean (Validation)"
        
        # Show legend only once for each unique entry
        # Training mean: show in first training subplot (row 1, col 1)
        # Validation mean: show in first validation subplot (row 2, col 1)
        show_mean_in_legend = False
        if row == 1 and col == 1 and data_type in ["loss", "accuracy"]:
            show_mean_in_legend = True  # Training mean in first subplot
        elif row == 2 and col == 1 and data_type in ["val_loss", "val_acc"]:
            show_mean_in_legend = True  # Validation mean in first validation subplot
        
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=data_mean,
                mode="lines",
                name=mean_name,
                line=dict(color=line_color, width=3.5),  # Thicker mean lines
                showlegend=show_mean_in_legend,
                legendgroup=mean_name,
            ),
            row=row, col=col
        )
        
        # Add individual run lines for all runs
        # Show runs in legend only in first subplot
        show_runs_in_legend = (row == 1 and col == 1)
        for run_id, data in enumerate(data_per_run):
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=data,
                    mode="lines",
                    name=f"Run {run_id + 1}",
                    line=dict(color=colors[run_id % len(colors)], width=1.5),
                    opacity=0.4,
                    showlegend=show_runs_in_legend,  # Show runs only in first subplot
                    legendgroup=f"run_{run_id + 1}",
                ),
                row=row, col=col
            )
    
    if has_stage2:
        # Row 1: Stage 1 Training Loss | Stage 1 Training Accuracy
        if epochs1:
            add_plot_to_subplot(1, 1, epochs1, train_losses_per_run1, "loss", 1)
            add_plot_to_subplot(1, 2, epochs1, train_accs_per_run1, "accuracy", 1)
        
        # Row 2: Stage 1 Validation Loss | Stage 1 Validation Accuracy
        if epochs1:
            add_plot_to_subplot(2, 1, epochs1, val_losses_per_run1, "val_loss", 1)
            add_plot_to_subplot(2, 2, epochs1, val_accs_per_run1, "val_acc", 1)
        
        # Row 3: Stage 2 Training Loss | Stage 2 Training Accuracy
        if epochs2:
            add_plot_to_subplot(3, 1, epochs2, train_losses_per_run2, "loss", 2)
            add_plot_to_subplot(3, 2, epochs2, train_accs_per_run2, "accuracy", 2)
        
        # Row 4: Stage 2 Validation Loss | Stage 2 Validation Accuracy
        if epochs2:
            add_plot_to_subplot(4, 1, epochs2, val_losses_per_run2, "val_loss", 2)
            add_plot_to_subplot(4, 2, epochs2, val_accs_per_run2, "val_acc", 2)
        
        # Update axes for 4x2 grid
        for row in range(1, 5):
            for col in range(1, 3):
                fig.update_xaxes(title_text="Epoch", row=row, col=col, title_font=dict(size=14), tickfont=dict(size=12))
                if col == 1:  # Left column: Loss
                    fig.update_yaxes(title_text="Loss", row=row, col=col, range=[0, None], title_font=dict(size=14), tickfont=dict(size=12))
                else:  # Right column: Accuracy
                    fig.update_yaxes(title_text="Accuracy", row=row, col=col, range=[0, 1.05], title_font=dict(size=14), tickfont=dict(size=12))
    else:
        # Single stage: Row 1: Training Loss | Training Accuracy, Row 2: Validation Loss | Validation Accuracy
        if epochs1:
            add_plot_to_subplot(1, 1, epochs1, train_losses_per_run1, "loss", 1)
            add_plot_to_subplot(1, 2, epochs1, train_accs_per_run1, "accuracy", 1)
            add_plot_to_subplot(2, 1, epochs1, val_losses_per_run1, "val_loss", 1)
            add_plot_to_subplot(2, 2, epochs1, val_accs_per_run1, "val_acc", 1)
        
        # Update axes for 2x2 grid
        for row in range(1, 3):
            for col in range(1, 3):
                fig.update_xaxes(
                    title_text="Epoch", 
                    row=row, col=col, 
                    title_font=dict(size=14), 
                    tickfont=dict(size=12),
                    scaleanchor=None,  # Don't force aspect ratio on x-axis
                )
                if col == 1:  # Left column: Loss
                    fig.update_yaxes(
                        title_text="Loss", 
                        row=row, col=col, 
                        range=[0, None], 
                        title_font=dict(size=14), 
                        tickfont=dict(size=12),
                    )
                else:  # Right column: Accuracy
                    fig.update_yaxes(
                        title_text="Accuracy", 
                        row=row, col=col, 
                        range=[0, 1.05], 
                        title_font=dict(size=14), 
                        tickfont=dict(size=12),
                    )
    
    # Determine architecture info and title
    arch = summary.get("architecture", {})
    num_params = summary.get("num_parameters", 0)
    
    # Check if it's an LSTM (no num_heads) or Transformer
    is_lstm = "num_heads" not in arch
    
    if is_lstm:
        num_layers = arch.get("num_layers", 1)
        d_model = arch.get("d_model", 64)
        if num_params:
            params_k = round(num_params / 1000)
            arch_subtitle = f"({num_layers} layers, d_model = {d_model}), ~{params_k}k parameters."
        else:
            arch_subtitle = f"({num_layers} layers, d_model = {d_model})"
        title_text = f"LSTMs: Training and Validation Metrics<br><sub>{arch_subtitle}</sub>"
    else:
        # Use provided architecture_info or default
        if architecture_info:
            arch_subtitle = architecture_info.replace("Transformer: ", "")
        else:
            arch_subtitle = "(2 layers, 4 heads, d_model = 64, d_ffn = 64), ~52k parameters."
        title_text = f"Transformers: Training and Validation Metrics<br><sub>{arch_subtitle}</sub>"
    
    # Set figure size: half size for single stage, full size for two stages
    if has_stage2:
        fig_height = 1600
        fig_width = 1400
    else:
        # Use square figure with proper aspect ratio
        fig_height = 1000
        fig_width = 1000
    
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=24, family="Arial Black"),
            x=0.5,  # Center the title
            xanchor="center",
            y=0.98,
        ),
        height=fig_height,
        width=fig_width,
        template="plotly_white",
        legend=dict(
            font=dict(size=16),
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
            tracegroupgap=2,  # Smaller spacing between legend entries
            itemwidth=30,  # Make legend items more compact
        ),
    )
    
    # For single-stage, ensure clean layout with proper margins
    if not has_stage2:
        # Let Plotly handle the layout automatically with the spacing we set
        # Just ensure margins are reasonable
        fig.update_layout(
            margin=dict(l=60, r=60, t=100, b=60),
        )
    
    # Save
    prefix = "transformer_" if architecture_info and "layers" in architecture_info else ""
    html_path = output_dir / f"{prefix}combined_stages.html"
    png_path = output_dir / f"{prefix}combined_stages.png"
    
    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path), width=fig_width, height=fig_height, scale=2)
        print(f"✓ Saved combined stages plot to: {html_path} and {png_path}")
    except Exception as e:
        print(f"✓ Saved combined stages plot to: {html_path} (PNG export failed: {e})")


def plot_query_length_extrapolation(summary: Dict, output_path: Path, architecture_info: str | None = None):
    """Plot line graph with error bars for query length extrapolation."""
    results = extract_query_length_results(summary)
    
    # Determine architecture info from summary if not provided
    if architecture_info is None:
        arch = summary.get("architecture", {})
        num_params = summary.get("num_parameters", 0)
        is_lstm = "num_heads" not in arch
        
        if is_lstm:
            num_layers = arch.get("num_layers", 1)
            d_model = arch.get("d_model", 64)
            if num_params:
                params_k = round(num_params / 1000)
                architecture_info = f"({num_layers} layers, d_model = {d_model}), ~{params_k}k parameters."
            else:
                architecture_info = f"({num_layers} layers, d_model = {d_model})"
        else:
            # Transformer case
            num_heads = arch.get("num_heads", 4)
            num_layers = arch.get("num_layers", 2)
            d_model = arch.get("d_model", 64)
            d_ff = arch.get("d_ff", 64)
            if num_params:
                params_k = round(num_params / 1000)
                architecture_info = f"(2 layers, 4 heads, d_model = {d_model}, d_ffn = {d_ff}), ~{params_k}k parameters."
            else:
                architecture_info = f"({num_layers} layers, {num_heads} heads, d_model = {d_model}, d_ffn = {d_ff})"
    
    # Get initial baseline accuracy from the last stage (stage_1, which corresponds to query length 90)
    # The initial_test_accuracy is calculated after all curriculum stages using the last stage's dataset
    initial_baseline_acc = None
    if "baseline_accuracies" in summary:
        stage_baselines = summary["baseline_accuracies"].get("curriculum_stages", {})
        # Get the last stage (stage_1, which is the second stage with query_length=90)
        last_stage_key = f"stage_{len(stage_baselines) - 1}" if stage_baselines else None
        if last_stage_key and last_stage_key in stage_baselines:
            initial_baseline_acc = stage_baselines[last_stage_key].get("test_baseline_accuracy")
    
    # Get initial query length from the last curriculum stage (which is 90)
    # The initial_test_accuracy is calculated using the last stage's query length
    initial_query_len = 90  # Default
    if "curriculum_stages" in summary and len(summary["curriculum_stages"]) > 0:
        # Get the last stage (index -1), which has query_length=90
        initial_query_len = summary["curriculum_stages"][-1].get("query_length", 90)
    
    # Prepare data: 100, Query 200, Query 300, Query 400
    labels = []
    query_lengths = []  # Store actual query lengths for table
    test_means = []
    test_stds = []
    baseline_means = []
    
    # Add initial test accuracy (from stage 1, query length 90)
    if results["initial_test_accs"]:
        labels.append(str(initial_query_len))
        query_lengths.append(initial_query_len)
        test_means.append(np.mean(results["initial_test_accs"]))
        test_stds.append(np.std(results["initial_test_accs"]))
        baseline_means.append(initial_baseline_acc)
    
    # Add query length results
    for query_len in sorted(results["query_lengths"]):
        labels.append(str(query_len))
        query_lengths.append(query_len)
        test_means.append(np.mean(results["test_accs"][query_len]))
        test_stds.append(np.std(results["test_accs"][query_len]))
        
        if query_len in results["baseline_accs"] and results["baseline_accs"][query_len]:
            baseline_means.append(np.mean(results["baseline_accs"][query_len]))
        else:
            baseline_means.append(None)
    
    # Convert to percentages for display
    test_means_pct = [m * 100 for m in test_means]
    test_stds_pct = [s * 100 for s in test_stds]
    baseline_means_pct = [b * 100 if b is not None else None for b in baseline_means]
    
    # Create figure with line graph and error bars
    fig = go.Figure()
    
    # Plot test accuracy line with error bars
    # Create upper and lower bounds for error bars
    upper_bounds = [m + s for m, s in zip(test_means_pct, test_stds_pct)]
    lower_bounds = [m - s for m, s in zip(test_means_pct, test_stds_pct)]
    
    # Add shaded region for error bars (mean ± std)
    fig.add_trace(go.Scatter(
        x=query_lengths + query_lengths[::-1],
        y=upper_bounds + lower_bounds[::-1],
        fill="toself",
        fillcolor="rgba(31, 119, 180, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
    ))
    
    # Add mean line
    fig.add_trace(go.Scatter(
        x=query_lengths,
        y=test_means_pct,
        mode="lines",
        name="Model Test Accuracy",
        line=dict(color="steelblue", width=4),
        error_y=dict(
            type="data",
            array=test_stds_pct,
            visible=True,
            thickness=2,
        ),
    ))
    
    # Determine if it's LSTM or Transformer for title prefix
    arch = summary.get("architecture", {})
    is_lstm = "num_heads" not in arch
    prefix = "LSTMs: " if is_lstm else "Transformers: "
    
    fig.update_layout(
        title=dict(
            text=f"{prefix}Query Length Extrapolation Performance{('<br><sub>' + architecture_info + '</sub>') if architecture_info else ''}",
            font=dict(size=22, family="Arial Black"),
        ),
        xaxis_title="Query Length",
        yaxis_title="Accuracy (%)",
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=16),
            type="linear",  # Ensure numeric x-axis
        ),
        yaxis=dict(
            range=[0, 105],  # 0 to 105% to match 1.05
            title_font=dict(size=18),
            tickfont=dict(size=16),
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(font=dict(size=16)),
        width=1000,
        height=600,
    )
    
    # Save as HTML
    html_path = output_path.with_suffix(".html")
    fig.write_html(str(html_path))
    # Also save as PNG image
    try:
        png_path = html_path.with_suffix(".png")
        fig.write_image(str(png_path), width=1000, height=600, scale=2)
        print(f"✓ Saved query length extrapolation plot to: {html_path} and {png_path}")
    except Exception as e:
        print(f"✓ Saved query length extrapolation plot to: {html_path} (PNG export failed: {e})")
    
    # Create and save table
    create_query_length_table(
        query_lengths=query_lengths,
        test_means=test_means,
        test_stds=test_stds,
        baseline_means=baseline_means,
        output_path=output_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Plot ablation study results")
    parser.add_argument(
        "summary_json",
        type=str,
        help="Path to summary.json file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same directory as summary.json)",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = summary_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading summary from: {summary_path}")
    summary = load_summary(summary_path)

    # Extract architecture information if available
    architecture_info = None
    arch = summary.get("architecture", {})
    if arch:
        # Check if it's a transformer (has num_heads)
        if "num_heads" in arch:
            num_heads = arch.get("num_heads", 4)
            num_layers = arch.get("num_layers", 2)
            d_model = arch.get("d_model", 64)
            d_ff = arch.get("d_ff", 64)
            num_params = summary.get("num_parameters", 0)
            if num_params:
                # Round to nearest thousand
                params_k = round(num_params / 1000)
                architecture_info = f"Transformer: ({num_layers} layers, {num_heads} heads, d_model = {d_model}, d_ffn = {d_ff}) ~{params_k}k parameters"
            else:
                architecture_info = f"Transformer: ({num_layers} layers, {num_heads} heads, d_model = {d_model}, d_ffn = {d_ff})"

    # Count how many stages have data
    runs = summary.get("runs", [])
    total_stages = 0
    if runs:
        stage_metrics = runs[0].get("stage_metrics", [])
        total_stages = sum(1 for stage in stage_metrics if stage)  # Count non-empty stages

    # Plot Stage 1
    print("\nPlotting Curriculum Stage 1...")
    (
        epochs1,
        train_losses_per_run1,
        val_losses_per_run1,
        train_accs_per_run1,
        val_accs_per_run1,
    ) = extract_stage_metrics_per_run(summary, stage_idx=0)

    if epochs1:
        # Get stage configuration from summary
        stage1_config = summary.get("curriculum_stages", [{}])[0] if summary.get("curriculum_stages") else {}
        num_states1 = stage1_config.get("num_states")
        num_actions1 = stage1_config.get("max_actions_per_state")
        
        plot_stage_accuracy(
            epochs1,
            train_accs_per_run1,
            val_accs_per_run1,
            stage_num=1,
            output_dir=output_dir,
            num_states=num_states1,
            num_actions=num_actions1,
            total_stages=total_stages,
            architecture_info=architecture_info,
        )

        plot_stage_loss(
            epochs1,
            train_losses_per_run1,
            val_losses_per_run1,
            stage_num=1,
            output_dir=output_dir,
            num_states=num_states1,
            num_actions=num_actions1,
            total_stages=total_stages,
            architecture_info=architecture_info,
        )
    else:
        print("⚠ No data for Stage 1")

    # Plot Stage 2
    print("\nPlotting Curriculum Stage 2...")
    (
        epochs2,
        train_losses_per_run2,
        val_losses_per_run2,
        train_accs_per_run2,
        val_accs_per_run2,
    ) = extract_stage_metrics_per_run(summary, stage_idx=1)

    if epochs2:
        # Get stage configuration from summary
        stage2_config = summary.get("curriculum_stages", [{}])[1] if len(summary.get("curriculum_stages", [])) > 1 else {}
        num_states2 = stage2_config.get("num_states")
        num_actions2 = stage2_config.get("max_actions_per_state")
        
        plot_stage_accuracy(
            epochs2,
            train_accs_per_run2,
            val_accs_per_run2,
            stage_num=2,
            output_dir=output_dir,
            num_states=num_states2,
            num_actions=num_actions2,
            total_stages=total_stages,
            architecture_info=architecture_info,
        )

        plot_stage_loss(
            epochs2,
            train_losses_per_run2,
            val_losses_per_run2,
            stage_num=2,
            output_dir=output_dir,
            num_states=num_states2,
            num_actions=num_actions2,
            total_stages=total_stages,
            architecture_info=architecture_info,
        )
    else:
        print("⚠ No data for Stage 2")

    # Plot combined stages
    print("\nPlotting combined stages...")
    plot_combined_stages(
        summary, output_dir=output_dir, architecture_info=architecture_info
    )

    # Plot query length extrapolation
    print("\nPlotting query length extrapolation...")
    # Add transformer prefix if architecture info indicates transformer
    prefix = "transformer_" if architecture_info and "layers" in architecture_info else ""
    plot_query_length_extrapolation(
        summary, output_path=output_dir / f"{prefix}query_length_extrapolation.png", architecture_info=architecture_info
    )

    print(f"\n{'='*70}")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

