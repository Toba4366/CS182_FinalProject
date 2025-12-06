"""
Linear Probing Experiment for LSTM Models

Tests whether a frozen LSTM backbone can solve FSM ICL with only a trainable linear head.
This evaluates if the LSTM has learned useful representations that enable in-context learning
purely through a linear transformation.

Research Question: Can frozen LSTM representations enable ICL through linear probing alone?
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.moore_lstm import MooreLSTM, LSTMConfig
from src.datasets.moore_dataset import MooreICLDataset


class LSTMLinearProbe(nn.Module):
    """LSTM backbone (frozen) + trainable linear probe head"""
    
    def __init__(self, lstm_backbone: MooreLSTM, num_states: int):
        super().__init__()
        self.backbone = lstm_backbone
        self.num_states = num_states
        
        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Create trainable linear probe
        # Input dimension is d_model * num_directions from LSTM
        lstm_output_dim = self.backbone.config.d_model * self.backbone.num_directions
        self.probe_head = nn.Linear(lstm_output_dim, num_states, bias=True)
        
        print(f"✓ Created linear probe: {lstm_output_dim} → {num_states}")
        
    def forward(self, input_ids, targets=None, unknown_mask=None):
        """
        Forward pass with frozen LSTM + trainable probe
        
        Args:
            input_ids: (batch, seq_len) token IDs
            targets: (batch, seq_len) ground truth state IDs  
            unknown_mask: (batch, seq_len) boolean mask for query positions
        
        Returns:
            logits: (batch, seq_len, num_states)
            loss: scalar loss (if targets provided)
        """
        B, T = input_ids.shape
        
        # Get LSTM representations (frozen, no gradients)
        with torch.no_grad():
            # Embed tokens
            x = self.backbone.token_embedding(input_ids)  # (B, T, d_model)
            x = self.backbone.dropout(x)
            
            # LSTM forward pass
            lstm_output, _ = self.backbone.lstm(x)  # (B, T, d_model * num_directions)
        
        # Apply trainable probe head (gradients enabled)
        logits = self.probe_head(lstm_output)  # (B, T, num_states)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            mask = (
                unknown_mask
                if unknown_mask is not None
                else torch.ones_like(targets, dtype=torch.bool)
            )
            
            flat_mask = mask.view(-1)
            flat_logits = logits.view(-1, self.num_states)
            flat_targets = targets.view(-1)
            
            if flat_mask.any():
                loss = F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask])
            else:
                loss = torch.tensor(0.0, device=input_ids.device)
        
        return logits, loss
    
    def count_parameters(self):
        """Count trainable vs frozen parameters"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        probe_params = sum(p.numel() for p in self.probe_head.parameters())
        
        return {
            'backbone_params': backbone_params,
            'probe_params': probe_params,
            'total_params': backbone_params + probe_params,
            'trainable_params': probe_params,
            'frozen_params': backbone_params
        }


def evaluate_probe(model, dataloader, device):
    """Evaluate linear probe accuracy"""
    model.eval()
    total_correct = 0
    total_predictions = 0
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            
            logits, loss = model(input_ids, target_ids, loss_mask)
            
            if loss is not None:
                total_loss += loss.item()
                num_batches += 1
            
            # Calculate accuracy on unknown positions only
            predictions = logits.argmax(dim=-1)
            correct = (predictions == target_ids) & loss_mask
            total_correct += correct.sum().item()
            total_predictions += loss_mask.sum().item()
    
    accuracy = total_correct / total_predictions if total_predictions > 0 else 0.0
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return accuracy, avg_loss


def main():
    parser = argparse.ArgumentParser(description="LSTM Linear Probing Experiment")
    
    # Model checkpoint
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained LSTM checkpoint (.pt file)")
    
    # Data
    parser.add_argument("--data_path", type=str, 
                       default="data/icl_dataset.pt",
                       help="Path to ICL dataset")
    parser.add_argument("--batch_size", type=int, default=8)
    
    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results/frozen_layers_test")
    
    args = parser.parse_args()
    
    print("="*80)
    print("LSTM LINEAR PROBING EXPERIMENT")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Training epochs: {args.num_epochs}")
    print()
    
    device = torch.device(args.device)
    
    # Load checkpoint
    print("Loading LSTM checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Reconstruct LSTM config from checkpoint
    # Handle different checkpoint formats
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
        vocab_size = model_config['vocab_size']
        num_states = model_config['num_states']
        max_seq_len = model_config.get('max_seq_len', 800)
        d_model = model_config.get('d_model', 256)
        num_layers = model_config.get('num_layers', 2)
        dropout = model_config.get('dropout', 0.1)
        bidirectional = model_config.get('bidirectional', False)
    else:
        # Older format - extract from checkpoint directly
        vocab_size = checkpoint.get('vocab_size', 21)
        # Infer from model state dict
        state_dict = checkpoint['model_state_dict']
        # Get num_states from head weight shape
        num_states = state_dict['head.weight'].shape[0]
        # Get d_model from token_embed (not token_embedding)
        if 'token_embed.weight' in state_dict:
            d_model = state_dict['token_embed.weight'].shape[1]
        elif 'token_embedding.weight' in state_dict:
            d_model = state_dict['token_embedding.weight'].shape[1]
        else:
            d_model = 256  # default
        # Get num_layers from LSTM weights
        num_layers = sum(1 for k in state_dict.keys() if 'lstm.weight_ih_l' in k)
        # Check if bidirectional
        bidirectional = any('_reverse' in k for k in state_dict.keys())
        max_seq_len = 800
        dropout = 0.1
    
    lstm_config = LSTMConfig(
        vocab_size=vocab_size,
        num_states=num_states,
        max_seq_len=max_seq_len,
        d_model=d_model,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional
    )
    
    # Create LSTM backbone
    lstm_backbone = MooreLSTM(lstm_config)
    
    # Load state dict with key mapping for older checkpoints
    state_dict = checkpoint['model_state_dict']
    # Map old key names to new ones if needed
    key_mapping = {
        'token_embed.weight': 'token_embedding.weight',
    }
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = key_mapping.get(k, k)
        new_state_dict[new_key] = v
    
    lstm_backbone.load_state_dict(new_state_dict)
    lstm_backbone.to(device)
    lstm_backbone.eval()
    
    print(f"✓ Loaded LSTM backbone")
    print(f"  Vocab size: {lstm_config.vocab_size}")
    print(f"  Num states: {lstm_config.num_states}")
    print(f"  d_model: {lstm_config.d_model}")
    print(f"  Layers: {lstm_config.num_layers}")
    print(f"  Bidirectional: {lstm_config.bidirectional}")
    print()
    
    # Create linear probe
    probe_model = LSTMLinearProbe(lstm_backbone, lstm_config.num_states)
    probe_model.to(device)
    
    # Count parameters
    param_counts = probe_model.count_parameters()
    print("Parameter counts:")
    print(f"  Backbone (frozen): {param_counts['backbone_params']:,}")
    print(f"  Probe (trainable): {param_counts['probe_params']:,}")
    print(f"  Total: {param_counts['total_params']:,}")
    print(f"  Frozen percentage: {param_counts['frozen_params']/param_counts['total_params']*100:.1f}%")
    print()
    
    # Load dataset using the same approach as training
    print(f"Loading dataset...")
    from src.datasets.moore_dataset import ICLDatasetConfig, load_or_create_icl_samples
    
    dataset_cfg = ICLDatasetConfig(
        num_samples=10_000,
        max_seq_len=lstm_config.max_seq_len,
    )
    
    all_samples = load_or_create_icl_samples(dataset_cfg)
    
    train_indices = list(range(0, 6000))
    val_indices = list(range(6000, 8000))
    test_indices = list(range(8000, 10_000))
    
    # Create dataset splits
    train_dataset = MooreICLDataset(
        train_indices,
        all_samples,
        dataset_cfg.traj_sampler_config,
        dataset_cfg.max_seq_len,
    )
    
    val_dataset = MooreICLDataset(
        val_indices,
        all_samples,
        dataset_cfg.traj_sampler_config,
        dataset_cfg.max_seq_len,
    )
    
    test_dataset = MooreICLDataset(
        test_indices,
        all_samples,
        dataset_cfg.traj_sampler_config,
        dataset_cfg.max_seq_len,
    )
    
    # Create data collator
    from src.training.lstm_trainer import ICLDataCollator
    collator = ICLDataCollator(pad_token_id=train_dataset.pad_token)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    
    print(f"✓ Dataset loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    print()
    
    # Setup optimizer (only for probe parameters)
    optimizer = torch.optim.AdamW(probe_model.probe_head.parameters(), lr=args.learning_rate)
    
    # Training loop
    print("Starting linear probe training...")
    print("-"*80)
    
    best_val_acc = 0.0
    history = {
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': []
    }
    
    for epoch in range(args.num_epochs):
        # Training
        probe_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            
            optimizer.zero_grad()
            logits, loss = probe_model(input_ids, target_ids, loss_mask)
            
            if loss is not None:
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy
                predictions = logits.argmax(dim=-1)
                correct = (predictions == target_ids) & loss_mask
                train_correct += correct.sum().item()
                train_total += loss_mask.sum().item()
        
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        train_loss = train_loss / len(train_loader)
        
        # Validation
        val_acc, val_loss = evaluate_probe(probe_model, val_loader, device)
        
        # Save history
        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['val_losses'].append(val_loss)
        history['val_accs'].append(val_acc)
        
        print(f"Epoch {epoch+1:2d}/{args.num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    print("-"*80)
    print(f"Training complete! Best val accuracy: {best_val_acc:.4f}")
    print()
    
    # Final test evaluation
    print("Evaluating on test set...")
    test_acc, test_loss = evaluate_probe(probe_model, test_loader, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print()
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'experiment': 'lstm_linear_probe',
        'timestamp': timestamp,
        'checkpoint': args.checkpoint,
        'model_config': {
            'vocab_size': lstm_config.vocab_size,
            'num_states': lstm_config.num_states,
            'd_model': lstm_config.d_model,
            'num_layers': lstm_config.num_layers,
            'bidirectional': lstm_config.bidirectional
        },
        'training_config': {
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'num_epochs': args.num_epochs
        },
        'parameter_counts': param_counts,
        'training_history': history,
        'final_results': {
            'best_val_accuracy': best_val_acc,
            'test_accuracy': test_acc,
            'test_loss': test_loss
        }
    }
    
    results_file = output_dir / f"lstm_linear_probe_{timestamp}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {results_file}")
    
    # Print summary
    print()
    print("="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Backbone: LSTM ({param_counts['backbone_params']:,} frozen parameters)")
    print(f"Probe: Linear ({param_counts['probe_params']:,} trainable parameters)")
    print(f"Training epochs: {args.num_epochs}")
    print(f"Best val accuracy: {best_val_acc:.2%}")
    print(f"Test accuracy: {test_acc:.2%}")
    print()
    print("Interpretation:")
    if test_acc > 0.9:
        print("  ✓ Excellent! Frozen LSTM representations enable strong ICL via linear probing")
    elif test_acc > 0.7:
        print("  ✓ Good! Linear probe can leverage LSTM representations for ICL")
    elif test_acc > 0.5:
        print("  ~ Moderate. LSTM learned some useful features but linear probing is limited")
    elif test_acc > 0.2:
        print("  ⚠ Weak performance. LSTM representations may not be linearly separable for ICL")
    else:
        print("  ✗ Poor performance. Frozen LSTM cannot enable ICL through linear probing alone")
    print("="*80)


if __name__ == "__main__":
    main()
