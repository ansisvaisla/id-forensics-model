"""Train Stage 2 — Presentation Attack classifier (EfficientNet-B0).

3 classes: screen (0), printout (1), live (2).

Usage:
    python scripts/training/train_stage2_screen.py
    python scripts/training/train_stage2_screen.py --epochs 40 --device cuda

Output: models/stage2_screen/best.pt
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "yolo" / "screen"
OUTPUT_DIR = PROJECT_ROOT / "models" / "stage2_screen"

_IMG_SIZE = 224
_BATCH = 32
_LR = 1e-4
_PATIENCE = 10
_NUM_CLASSES = 3
_CLASS_NAMES = ("screen", "printout", "live")


def _build_dataset(split: str, augment: bool):
    """Build a PyTorch Dataset from screen/labels/{split}/*.txt + images."""
    import torch  # type: ignore
    from torch.utils.data import Dataset  # type: ignore
    import torchvision.transforms as T  # type: ignore
    import cv2  # type: ignore

    img_dir = DATA_DIR / "images" / split
    lbl_dir = DATA_DIR / "labels" / split

    files = sorted(img_dir.glob("*.jpg"))

    base_transforms = [
        T.ToPILImage(),
        T.Resize((_IMG_SIZE, _IMG_SIZE)),
    ]
    if augment:
        aug = [
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.RandomRotation(10),
        ]
    else:
        aug = []

    transform = T.Compose(base_transforms + aug + [
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class ScreenDataset(Dataset):
        def __init__(self) -> None:
            self.samples: list[tuple[Path, int]] = []
            for img_path in files:
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                if not lbl_path.is_file():
                    continue
                label = int(lbl_path.read_text(encoding="utf-8").strip())
                self.samples.append((img_path, label))

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            img_path, label = self.samples[idx]
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return transform(img), label

    return ScreenDataset()


def _build_model(pretrained: bool):
    """Build EfficientNet-B0 with 3-class head."""
    import torch.nn as nn  # type: ignore
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0  # type: ignore

    if pretrained:
        try:
            print("Loading ImageNet weights from torchvision...")
            model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        except Exception as exc:
            print(f"WARNING: pretrained download failed ({exc}). Training from scratch.")
            model = efficientnet_b0(weights=None)
    else:
        print("Training from scratch (--no-pretrained).")
        model = efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, _NUM_CLASSES)
    return model


def _compute_class_weights(samples: list[tuple[Path, int]], device) -> "torch.Tensor":
    """Inverse-frequency weights to handle class imbalance (printout is rare)."""
    import torch  # type: ignore

    counts = Counter(lbl for _, lbl in samples)
    total = len(samples)
    weights = [total / (len(counts) * counts.get(c, 1)) for c in range(_NUM_CLASSES)]
    print("Class distribution:")
    for i, name in enumerate(_CLASS_NAMES):
        print(f"  {name:10s}: {counts.get(i, 0):4d}  weight={weights[i]:.3f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Stage 2 presentation attack classifier")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=_BATCH)
    parser.add_argument("--lr", type=float, default=_LR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip ImageNet weight download (use if torchvision CDN is blocked)",
    )
    args = parser.parse_args()

    # Normalise bare digit device string ("0" -> "cuda:0")
    raw_dev = args.device.strip()
    if raw_dev.isdigit():
        raw_dev = f"cuda:{raw_dev}"

    if not (DATA_DIR / "images" / "train").is_dir():
        print(f"Train split not found: {DATA_DIR}", file=sys.stderr)
        print("Run scripts/convert_labels_to_yolo.py + split_yolo_dataset.py first.", file=sys.stderr)
        return 1

    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        from torch.utils.data import DataLoader  # type: ignore
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Run: pip install torch torchvision", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(raw_dev)

    train_ds = _build_dataset("train", augment=True)
    val_ds = _build_dataset("val", augment=False)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    weight = _compute_class_weights(train_ds.samples, device)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = _build_model(pretrained=not args.no_pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total else 0.0
        print(f"Epoch {epoch:3d}  loss={total_loss / len(train_loader):.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(OUTPUT_DIR / "best.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= _PATIENCE:
                print(f"Early stop at epoch {epoch} (patience={_PATIENCE})")
                break

    torch.save(model.state_dict(), str(OUTPUT_DIR / "last.pt"))
    print(f"\nBest val acc: {best_val_acc:.4f}")
    print(f"Weights: {OUTPUT_DIR / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
