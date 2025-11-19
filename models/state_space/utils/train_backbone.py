#!/usr/bin/env python3
"""
Train S4 or Mamba backbone on FSM next-token prediction.
Loads train_tokens.json / val_tokens.json directly.
"""

import json
import torch
import argparse
from torch.utils.data import Dataset, DataLoader

# Correct imports (your repo structure)
from models.state_space.s4 import S4LM
# from models.state_space.mamba import MambaLM
from utils.training import (
    create_optimizer,
    create_warmup_cosine_scheduler,
    train_step_generic
)

# ------------------------------
# Dataset
# ------------------------------
class FSMDataset(Dataset):
    """
    Loads tokens from {split}_tokens.json
    Produces (input, target) pairs for next-token prediction.
    """
    def __init__(self, json_file, max_length=256):
        with open(json_file, "r") as f:
            data = json.load(f)

        self.samples = data["samples"]
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]["tokens"]

        # Truncate or pad the token sequence
        tokens = tokens[:self.max_length]
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))

        # Create next-token prediction (language modeling)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        targets   = torch.roll(input_ids, shifts=-1)
        

        return input_ids, targets


# ------------------------------
# Training Loop
# ------------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset
    train_ds = FSMDataset(f"{args.data_dir}/train_tokens.json", max_length=args.context)
    val_ds   = FSMDataset(f"{args.data_dir}/val_tokens.json",   max_length=args.context)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch)

    # Model choice
    if args.model == "s4":
        model = S4LM(
            vocab_size=args.vocab,
            d_model=args.d_model,
            n_layers=args.layers,
            max_seq_len=args.context
        )
    else:
        model = MambaLM(
            vocab_size=args.vocab,
            d_model=args.d_model,
            n_layers=args.layers,
            max_seq_len=args.context
        )

    model = model.to(device)

    # Optimizer/scheduler
    optim = create_optimizer(model, lr=args.lr)
    sched = create_warmup_cosine_scheduler(optim, warmup_steps=100, total_steps=args.steps)

    print(f"Training {args.model.upper()} backbone for {args.steps} steps...")

    step = 0
    while step < args.steps:
        for batch in train_loader:
            loss, _ = train_step_generic(
                model=model,
                optimizer=optim,
                scheduler=sched,
                batch=batch,
                model_type="transformer",
                device=device
            )
            step += 1

            if step % 100 == 0:
                print(f"[step {step}] train loss = {loss:.4f}")

            if step >= args.steps:
                break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": args.vocab,
            "d_model": args.d_model,
            "layers": args.layers,
        },
        args.out
    )
    print(f"Saved backbone checkpoint to {args.out}")


# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["s4", "mamba"], default="s4")
    parser.add_argument("--data-dir", default="data/full_dataset_json")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--out", default="backbone.pt")

    args = parser.parse_args()
    train(args)
