"""
Visualization utilities for training curves, FSM diagrams, and attention patterns.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import networkx as nx

from ..fsm.moore_machine import MooreMachine

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_curves(train_losses: List[float],
                        val_losses: Optional[List[float]] = None,
                        save_path: Optional[str] = None,
                        title: str = "Training Curves") -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses (optional)
        save_path: Path to save the plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', alpha=0.8)
    if val_losses:
        # Val losses are recorded less frequently, so interpolate x-axis
        val_x = np.linspace(0, len(train_losses) - 1, len(val_losses))
        ax1.plot(val_x, val_losses, label='Validation Loss', alpha=0.8)
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot log-scale losses
    ax2.semilogy(train_losses, label='Training Loss', alpha=0.8)
    if val_losses:
        ax2.semilogy(val_x, val_losses, label='Validation Loss', alpha=0.8)
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Loss Curves (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_fsm_diagram(fsm: MooreMachine,
                    save_path: Optional[str] = None,
                    title: str = "Moore Machine") -> plt.Figure:
    """
    Plot a visual diagram of the Moore machine.
    
    Args:
        fsm: Moore machine to visualize
        save_path: Path to save the plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes (states)
    for state in fsm.states:
        output = fsm.outputs.get(state, "")
        G.add_node(state, label=f"S{state}\n{output}")
    
    # Add edges (transitions)
    for (from_state, action), to_state in fsm.transitions.items():
        if G.has_edge(from_state, to_state):
            # Multiple transitions between same states
            existing_label = G[from_state][to_state].get('label', '')
            new_label = f"{existing_label}, A{action}"
        else:
            new_label = f"A{action}"
        
        G.add_edge(from_state, to_state, label=new_label)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw nodes
    node_colors = ['lightcoral' if state == fsm.initial_state else 'lightblue' 
                  for state in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=2000, alpha=0.8, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20, 
                          arrowstyle='->', ax=ax)
    
    # Draw labels
    node_labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, node_labels, font_size=10, ax=ax)
    
    # Draw edge labels
    edge_labels = {(u, v): data['label'] for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
    
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_attention(attention_weights: torch.Tensor,
                       tokens: List[str],
                       layer: int = -1,
                       head: int = 0,
                       save_path: Optional[str] = None,
                       title: str = "Attention Weights") -> plt.Figure:
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weights tensor (layers, heads, seq_len, seq_len)
        tokens: List of token strings
        layer: Which layer to visualize (-1 for last layer)
        head: Which attention head to visualize
        save_path: Path to save the plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    if attention_weights.dim() != 4:
        raise ValueError("Expected 4D attention weights: (layers, heads, seq_len, seq_len)")
    
    # Extract attention for specified layer and head
    attn = attention_weights[layer, head].detach().cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(attn, 
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='Blues',
                ax=ax,
                cbar_kws={'label': 'Attention Weight'})
    
    ax.set_title(f"{title} - Layer {layer}, Head {head}")
    ax.set_xlabel('Key Tokens')
    ax.set_ylabel('Query Tokens')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_accuracy_by_complexity(complexity_results: Dict[str, Dict],
                               save_path: Optional[str] = None,
                               title: str = "Performance by Complexity") -> plt.Figure:
    """
    Plot performance metrics grouped by complexity factors.
    
    Args:
        complexity_results: Results from analyze_performance_by_complexity
        save_path: Path to save the plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    n_categories = len(complexity_results)
    fig, axes = plt.subplots(1, n_categories, figsize=(5 * n_categories, 5))
    
    if n_categories == 1:
        axes = [axes]
    
    for i, (category, groups) in enumerate(complexity_results.items()):
        ax = axes[i]
        
        group_names = []
        accuracies = []
        exact_matches = []
        
        for group_name, metrics in groups.items():
            group_names.append(str(group_name))
            accuracies.append(metrics['accuracy'])
            exact_matches.append(metrics['exact_match_rate'])
        
        x = np.arange(len(group_names))
        width = 0.35
        
        ax.bar(x - width/2, accuracies, width, label='Token Accuracy', alpha=0.8)
        ax.bar(x + width/2, exact_matches, width, label='Exact Match Rate', alpha=0.8)
        
        ax.set_xlabel(category.replace('_', ' ').title())
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Performance by {category.replace("_", " ").title()}')
        ax.set_xticks(x)
        ax.set_xticklabels(group_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for j, (acc, em) in enumerate(zip(accuracies, exact_matches)):
            ax.text(j - width/2, acc + 0.01, f'{acc:.3f}', 
                   ha='center', va='bottom', fontsize=8)
            ax.text(j + width/2, em + 0.01, f'{em:.3f}', 
                   ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_sequence_prediction_examples(model_outputs: List[Tuple[List[str], List[str]]],
                                    save_path: Optional[str] = None,
                                    max_examples: int = 5,
                                    title: str = "Prediction Examples") -> plt.Figure:
    """
    Plot examples of sequence predictions vs ground truth.
    
    Args:
        model_outputs: List of (predicted_sequence, target_sequence) tuples
        save_path: Path to save the plot
        max_examples: Maximum number of examples to show
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    n_examples = min(len(model_outputs), max_examples)
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 2 * n_examples))
    
    if n_examples == 1:
        axes = [axes]
    
    for i, (predicted, target) in enumerate(model_outputs[:n_examples]):
        ax = axes[i]
        
        max_len = max(len(predicted), len(target))
        x = np.arange(max_len)
        
        # Pad sequences to same length for visualization
        pred_padded = predicted + [''] * (max_len - len(predicted))
        target_padded = target + [''] * (max_len - len(target))
        
        # Create comparison
        matches = [p == t for p, t in zip(pred_padded, target_padded)]
        colors = ['green' if match else 'red' for match in matches]
        
        # Plot as text
        for j, (p, t, color) in enumerate(zip(pred_padded, target_padded, colors)):
            ax.text(j, 1, f"P: {p}", ha='center', va='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
            ax.text(j, 0, f"T: {t}", ha='center', va='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.3))
        
        ax.set_xlim(-0.5, max_len - 0.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xticks(range(max_len))
        ax.set_xticklabels([f'T{j}' for j in range(max_len)])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Target', 'Predicted'])
        ax.set_title(f'Example {i+1} - Accuracy: {sum(matches)/len(matches):.2f}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig