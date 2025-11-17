import torch
import torch.nn as nn
from torch.optim import AdamW

from linear_probe_s4 import S4LinearProbe
from linear_probe_mamba import MambaLinearProbe


def train_probe(
    model,
    dataloader,
    lr=1e-3,
    device="cuda",
    max_steps=2000,
):
    model.to(device)
    optimizer = AdamW(model.head.parameters(), lr=lr)

    step = 0
    model.train()

    for batch in dataloader:
        input_ids, targets = batch
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        # Only head parameters require gradients
        optimizer.zero_grad()

        logits = model(input_ids)
        loss = nn.CrossEntropyLoss(ignore_index=-100)(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1)
        )
        loss.backward()
        optimizer.step()

        step += 1
        if step % 100 == 0:
            print(f"[step {step}] loss={loss.item():.4f}")

        if step >= max_steps:
            break

    return model
