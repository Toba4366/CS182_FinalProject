"""
Entry point for training the Moore LSTM in an ICL setting.
"""

from __future__ import annotations

import argparse
from torch.utils.data import DataLoader, random_split # type: ignore

from src.datasets.moore_dataset import (
    ICLDatasetConfig,
    MooreICLDataset,
    load_or_create_icl_samples,
)
from src.models.moore_lstm import MooreLSTM, LSTMConfig
from src.training.lstm_trainer import (
    ICLDataCollator,
    MooreLSTMTrainer,
    TrainingConfig,
    evaluate_lstm_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LSTM for Moore ICL")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-verbose", action="store_true", help="Disable verbose training output")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_cfg = ICLDatasetConfig(
        num_samples=max(args.num_samples, 10_000),
        max_seq_len=args.max_seq_len,
    )

    if dataset_cfg.num_samples < 10_000:
        raise ValueError("Dataset must contain at least 10,000 samples.")

    all_samples = load_or_create_icl_samples(dataset_cfg)

    train_indices = list(range(0, 6000))
    val_indices = list(range(6000, 8000))
    test_indices = list(range(8000, 10_000))

    train_dataset = MooreICLDataset(
        train_indices,
        all_samples,
        dataset_cfg.sampler_config,
        dataset_cfg.max_seq_len,
    )
    val_dataset = MooreICLDataset(
        val_indices,
        all_samples,
        dataset_cfg.sampler_config,
        dataset_cfg.max_seq_len,
    )
    test_dataset = MooreICLDataset(
        test_indices,
        all_samples,
        dataset_cfg.sampler_config,
        dataset_cfg.max_seq_len,
    )

    model_cfg = LSTMConfig(
        vocab_size=train_dataset.vocab_size,
        num_states=train_dataset.num_states,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )
    model = MooreLSTM(model_cfg)

    collator = ICLDataCollator(pad_token_id=train_dataset.pad_token)
    trainer_cfg = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        device=args.device,
        verbose=not args.no_verbose,  # verbose=True by default, unless --no-verbose
    )
    trainer = MooreLSTMTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collator=collator,
        config=trainer_cfg,
    )
    trainer.train()

    print("Evaluating on validation set...")
    val_acc = evaluate_lstm_model(trainer.model, trainer.val_loader, trainer.device)
    print(f"Validation Accuracy: {val_acc.item():.4f}")

    print("Evaluating on test set...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=trainer_cfg.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    test_acc = evaluate_lstm_model(trainer.model, test_loader, trainer.device)
    print(f"Test Accuracy: {test_acc.item():.4f}")


if __name__ == "__main__":
    main()