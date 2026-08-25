"""PAD-UFES-20 non-private training loop and shared evaluation."""

import torch
import torch.nn as nn
import torch.optim as optim


# --------------------------------------------------------------------------- #
# Standard training (mode = sgd)
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x_b, y_b in loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y_b.size(0)
        correct += (logits.argmax(1) == y_b).sum().item()
        total += y_b.size(0)
    return total_loss / total, correct / total


def run_training(model, train_loader, val_loader, device, epochs=100, lr=1e-3, label='Model'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"[{label}] Epoch {epoch:>3}/{epochs}  "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
    return {'train_loss': tr_loss, 'train_acc': tr_acc, 'val_loss': va_loss, 'val_acc': va_acc}


# --------------------------------------------------------------------------- #
# Shared evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x_b, y_b in loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        logits = model(x_b)
        total_loss += criterion(logits, y_b).item() * y_b.size(0)
        correct += (logits.argmax(1) == y_b).sum().item()
        total += y_b.size(0)
    return total_loss / total, correct / total
