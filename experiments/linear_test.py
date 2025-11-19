import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Import backbone architectures
from models.state_space.s4 import S4LM
from models.state_space.mamba import MambaLM

# Training utilities
from utils.training import (
    create_optimizer,
    train_step_generic
)

# -----------------------------
# Dataset
# -----------------------------
class TokenDataset(Dataset):
    def __init__(self, token_path):
        with open(token_path, "r") as f:
            data = json.load(f)
        self.inputs = torch.tensor(data["input_ids"], dtype=torch.long)
        self.targets = torch.tensor(data["targets"], dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# -----------------------------
# Linear Probe Model
# -----------------------------
class LinearProbe(nn.Module):
    def __init__(self, backbone, d_model, num_states):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head = nn.Linear(d_model, num_states)

    def forward(self, x, targets=None):
        h = self.backbone(x)              # (B, T, D)
        logits = self.head(h)             # (B, T, num_states)

        if targets is None:
            return logits, None

        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)),
                                     targets.view(-1))
        return logits, loss


# -----------------------------
# Main experiment
# -----------------------------
def run_experiment(args):
    device = torch.device(args.device)

    # Load vocab to get number of prediction classes
    with open(f"{args.dataset}/vocab.json") as f:
        vocab = json.load(f)
    num_states = len(vocab)

    # Load backbone
    if args.model == "s4":
        backbone = S4LM(
            vocab_size=num_states,
            d_model=args.d_model,
            n_layers=args.layers
        )
    elif args.model == "mamba":
        backbone = MambaLM(
            vocab_size=num_states,
            d_model=args.d_model,
            n_layers=args.layers
        )
    else:
        raise ValueError("Unknown model name")

    # Load backbone weights
    ckpt = torch.load(args.backbone_ckpt, map_location=device)
    backbone.load_state_dict(ckpt["model_state_dict"])
    backbone.to(device)

    # Create probe
    probe = LinearProbe(
        backbone=backbone,
        d_model=args.d_model,
        num_states=num_states
    ).to(device)

    # Dataset
    train_ds = TokenDataset(f"{args.dataset}/train_tokens.json")
    val_ds   = TokenDataset(f"{args.dataset}/val_tokens.json")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch)

    # Optimizer (only trains Linear layer!)
    optimizer = create_optimizer(probe.head, lr=args.lr, weight_decay=0.0)

    print("Starting linear probe training")

    for step, batch in enumerate(train_loader):
        if step >= args.steps:
            break

        loss, _ = train_step_generic(
            model=probe,
            optimizer=optimizer,
            scheduler=None,
            batch=batch,
            model_type="transformer",
            device=device
        )

        if step % 50 == 0:
            print(f"[step {step}] loss={loss:.4f}")

    # -----------------
    # Evaluate
    # -----------------
    print("\n Evaluating probe accuracy")
    correct = 0
    total = 0

    probe.eval()
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)

            logits, _ = probe(x)
            pred = logits.argmax(dim=-1)

            correct += (pred == y).sum().item()
            total += y.numel()

    acc = correct / total
    print(f"\n Probe accuracy: {acc:.4f}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="s4", choices=["s4", "mamba"])
    parser.add_argument("--dataset", type=str, default="data/full_dataset_json")
    parser.add_argument("--backbone_ckpt", type=str, required=True)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)

    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    run_experiment(args)
