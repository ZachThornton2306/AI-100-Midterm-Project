#!/usr/bin/env python3
"""
mnist_cnn.py

Trains a deep learning CNN on MNIST to solve a classification problem:
predicting the digit (0-9) from a 28x28 grayscale image.

How to run:
  python mnist_cnn.py --epochs 3
  python mnist_cnn.py --epochs 5 --batch-size 128 --lr 0.001

Outputs:
  - Prints training loss + validation accuracy each epoch
  - Saves best model to: ./mnist_cnn_best.pt
"""


from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# torchvision is the easiest way to load MNIST
from torchvision import datasets, transforms


def pick_device() -> torch.device:
    """
    Choose the best available device:
    - Apple Silicon GPU via MPS (Metal Performance Shaders), if available
    - CUDA GPU, if available
    - CPU otherwise
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MNIST_CNN(nn.Module):
    """
    A simple but effective CNN for MNIST:
      Conv -> Conv -> Pool -> Dropout -> Conv -> Pool -> FC -> Dropout -> FC(10)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 14x14 -> 14x14 after pool
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)   # 28x28 -> 14x14
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2)   # 14x14 -> 7x7

        x = torch.flatten(x, 1)              # (N, 128*7*7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                      # logits (N, 10)
        return x


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(n_batches, 1)


@dataclass(frozen=True)
class Config:
    epochs: int
    batch_size: int
    lr: float
    seed: int
    data_dir: Path
    save_path: Path
    val_fraction: float
    num_workers: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST (classification) using PyTorch.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--save-path", type=Path, default=Path("./mnist_cnn_best.pt"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    cfg = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        data_dir=args.data_dir,
        save_path=args.save_path,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
    )

    # Reproducibility
    torch.manual_seed(cfg.seed)

    device = pick_device()
    print(f"Using device: {device}")

    # MNIST transforms: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Download/load MNIST
    train_full = datasets.MNIST(root=str(cfg.data_dir), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=str(cfg.data_dir), train=False, download=True, transform=transform)

    # Split train into train/val
    val_size = int(len(train_full) * cfg.val_fraction)
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(
        train_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    # DataLoaders
    # Note: On macOS, num_workers>0 can sometimes be finicky; set to 0 if you see issues.
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
    )

    model = MNIST_CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = accuracy(model, val_loader, device)

        print(f"Epoch {epoch:02d}/{cfg.epochs} | loss={avg_loss:.4f} | val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), cfg.save_path)
            print(f"  Saved new best model -> {cfg.save_path} (val_acc={best_val_acc*100:.2f}%)")

    # Load best and evaluate on test set
    if cfg.save_path.exists():
        state_dict = torch.load(cfg.save_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)

    test_acc = accuracy(model, test_loader, device)
    print(f"Final TEST accuracy: {test_acc*100:.2f}%")


if __name__ == "__main__":
    # Helps avoid some multiprocessing quirks on macOS with DataLoader workers
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()