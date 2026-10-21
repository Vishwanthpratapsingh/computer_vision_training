"""Module 1: MLP and CNN on CIFAR-10 with a hand-written training loop.

Usage:
    pip install torch torchvision matplotlib wandb
    python train.py --model mlp --epochs 10
    python train.py --model cnn --epochs 15 --augment --wandb

No pretrained models. Saves curves_<model>.png and prints final test accuracy.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

dev = "cuda" if torch.cuda.is_available() else "cpu"


def get_loaders(batch_size, augment):
    norm = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    aug = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if augment else []
    train_tf = T.Compose(aug + [T.ToTensor(), norm])
    eval_tf = T.Compose([T.ToTensor(), norm])

    # Two views of the same training set so validation never gets augmentation
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(50000, generator=g).tolist()
    train_idx, val_idx = perm[:45000], perm[45000:]
    train_full = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=train_tf)
    val_full = torchvision.datasets.CIFAR10("data", train=True, download=True, transform=eval_tf)
    test_ds = torchvision.datasets.CIFAR10("data", train=False, download=True, transform=eval_tf)

    train_dl = DataLoader(Subset(train_full, train_idx), batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(Subset(val_full, val_idx), 256, num_workers=2)
    test_dl = DataLoader(test_ds, 256, num_workers=2)
    return train_dl, val_dl, test_dl


def make_mlp():
    # 2-layer MLP
    return nn.Sequential(nn.Flatten(), nn.Linear(3072, 512), nn.ReLU(), nn.Linear(512, 10))


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
        nn.MaxPool2d(2),
    )


def make_cnn():
    return nn.Sequential(
        conv_block(3, 32), conv_block(32, 64), conv_block(64, 128),  # 32 -> 16 -> 8 -> 4
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(128 * 4 * 4, 10),
    )


def run_epoch(model, loader, opt=None):
    training = opt is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)                  # 1. forward
            loss = F.cross_entropy(logits, y)  # 2. loss
            if training:
                opt.zero_grad()                # clear old gradients
                loss.backward()                # 3. backward (autograd)
                opt.step()                     # 4. optimizer step
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["mlp", "cnn"], default="cnn")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--wandb", action="store_true")
    a = p.parse_args()

    if a.wandb:
        import wandb
        wandb.init(project="module-1-ml-cv", name=a.model, config=vars(a))

    train_dl, val_dl, test_dl = get_loaders(a.batch_size, a.augment)
    model = (make_mlp() if a.model == "mlp" else make_cnn()).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for ep in range(1, a.epochs + 1):
        tl, ta = run_epoch(model, train_dl, opt)
        vl, va = run_epoch(model, val_dl)
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
        print(f"epoch {ep:02d} | train loss {tl:.3f} acc {ta:.3f} | val loss {vl:.3f} acc {va:.3f}")
        if a.wandb:
            wandb.log({"epoch": ep, "train_loss": tl, "val_loss": vl, "train_acc": ta, "val_acc": va})

    _, test_acc = run_epoch(model, test_dl)
    print(f"TEST ACCURACY: {test_acc:.3f}")
    if a.wandb:
        wandb.log({"test_acc": test_acc})

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val")
    ax[1].set_title(f"Accuracy (test: {test_acc:.3f})"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"curves_{a.model}.png", dpi=120)


if __name__ == "__main__":
    main()
