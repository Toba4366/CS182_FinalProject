"""
Experiment runner for Moore machine in-context learning experiments.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import yaml

# Add project root to path  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.fsm.moore_machine import MooreMachineGenerator
from src.training.dataset import MooreMachineDataset
from src.training.models import SimpleTransformer, TransformerConfig
from src.training.trainer import ICLTrainer


def setup_logging(log_file: Optional[str] = None):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_datasets(config: Dict[str, Any]) -> tuple:
    """Create train and validation datasets."""
    data_config = config['data']
    
    # Training dataset
    train_dataset = MooreMachineDataset(
        num_machines=data_config['train_machines'],
        examples_per_machine=data_config['examples_per_machine'],
        sequence_length=data_config['sequence_length'],
        test_sequence_length=data_config['test_sequence_length'],
        vocab_size=data_config['vocab_size'],
        seed=config['seed']
    )
    
    # Validation dataset (different seed)
    val_dataset = MooreMachineDataset(
        num_machines=data_config['val_machines'],
        examples_per_machine=data_config['examples_per_machine'],
        sequence_length=data_config['sequence_length'],
        test_sequence_length=data_config['test_sequence_length'],
        vocab_size=data_config['vocab_size'],
        seed=config['seed'] + 1000  # Different seed
    )
    
    return train_dataset, val_dataset


def create_model(config: Dict[str, Any], vocab_size: int) -> SimpleTransformer:
    """Create transformer model."""
    model_config = TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=config['model']['max_seq_len'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout']
    )
    
    return SimpleTransformer(model_config)


def run_experiment(config_path: str, output_dir: str):
    """Run a complete experiment."""
    # Load configuration
    config = load_config(config_path)
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(str(output_path / 'experiment.log'))
    logger = logging.getLogger(__name__)
    
    logger.info("Starting experiment")
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    # Save config to output directory
    with open(output_path / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_datasets(config)
    
    # Save dataset samples for inspection
    data_path = output_path / 'data'
    data_path.mkdir(exist_ok=True)
    train_dataset.save_to_disk(str(data_path / 'train_dataset.pkl'))
    val_dataset.save_to_disk(str(data_path / 'val_dataset.pkl'))
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config, train_dataset.vocab_size)
    logger.info(f"Model has {model.count_parameters():,} parameters")
    
    # Create trainer
    logger.info("Setting up trainer...")
    trainer = ICLTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        max_steps=config['training']['max_steps'],
        batch_size=config['training']['batch_size'],
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        gradient_clip_norm=config['training']['gradient_clip_norm'],
        save_dir=str(output_path / 'checkpoints'),
        use_wandb=config.get('use_wandb', False),
        wandb_project=config.get('wandb_project', 'moore-icl'),
        seed=config['seed']
    )
    
    # Train model
    logger.info("Starting training...")
    training_history = trainer.train()
    
    # Save training history
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info("Experiment completed!")
    logger.info(f"Results saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Moore machine ICL experiment')
    parser.add_argument('--config', '-c', required=True,
                       help='Path to experiment configuration file')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for experiment results')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Set GPU device if specified
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Run experiment
    run_experiment(args.config, args.output)


if __name__ == '__main__':
    main()