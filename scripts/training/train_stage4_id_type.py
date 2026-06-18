"""Train Stage 4 — ID Type Classifier (EfficientNet-B0, 7-class).

Uses ImageFolder structure built by convert_labels_to_yolo.py:
    data/id_type/
        train/<class>/<image>.jpg
        val/<class>/<image>.jpg
        test/<class>/<image>.jpg

Classes (by index):
    0 legacy          - old Kenyan national ID (blue chipboard)
    1 maisha          - new Kenyan national ID (Maisha Card)
    2 huduma          - Huduma Namba (limited rollout, rare)
    3 passport         - Kenyan or foreign passport bio-page
    4 driving_licence  - Kenyan driving licence
    5 foreign_document - other country national IDs
    6 unknown_id       - unrecognisable / unclear

Data split is produced by scripts/split_id_type_dataset.py.
The output weights file is models/stage4_id_type/best.pt.

Usage:
    python scripts/training/train_stage4_id_type.py
    python scripts/training/train_stage4_id_type.py --epochs 50 --device cuda
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "id_type"
OUTPUT_DIR = PROJECT_ROOT / "models" / "stage4_id_type"

CLASSES = (
    "legacy",
    "maisha",
    "huduma",
    "passport",
    "driving_licence",
    "foreign_document",
    "unknown_id",
)

_IMG_SIZE = 224
_BATCH = 32
_LR = 1e-4
_PATIENCE = 10


def _build_dataset(split: str, augment: bool):
    import cv2  # type: ignore
    import torch  # type: ignore
    import torchvision.transforms as T  # type: ignore
    from torch.utils.data import Dataset  # type: ignore

    split_dir = DATA_DIR / split

    base = [T.ToPILImage(), T.Resize((_IMG_SIZE, _IMG_SIZE))]
    aug = (
        [
            T.RandomHorizontalFlip(),
            T.RandomRotation(8),
            T.ColorJitter(brightness=0.3, contrast=0.25, saturation=0.15),
            T.RandomPerspective(distortion_scale=0.1, p=0.3),
        ]
        if augment
        else []
    )
    transform = T.Compose(
        base
        + aug
        + [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class IdTypeDataset(Dataset):
        def __init__(self) -> None:
            self.samples: list[tuple[Path, int]] = []
            for cls_idx, cls_name in enumerate(CLASSES):
                cls_dir = split_dir / cls_name
                if not cls_dir.is_dir():
                    continue
                for img_path in sorted(cls_dir.glob("*.jpg")):
                    self.samples.append((img_path, cls_idx))
                for img_path in sorted(cls_dir.glob("*.jpeg")):
                    self.samples.append((img_path, cls_idx))

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            img_path, label = self.samples[idx]
            img = cv2.imread(str(img_path))
            if img is None:
                raise OSError(f"Cannot read {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return transform(img), label

    return IdTypeDataset()


def _build_model(pretrained: bool, num_classes: int):
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
        model = efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _compute_class_weights(train_ds) -> "torch.Tensor":
    """Inverse-frequency class weighting to handle imbalanced classes."""
    import torch  # type: ignore
    from collections import Counter

    counts = Counter(lbl for _, lbl in train_ds.samples)
    total = len(train_ds)
    weights = []
    for i in range(len(CLASSES)):
        n = counts.get(i, 1)  # avoid div/0 for unseen classes
        weights.append(total / (len(CLASSES) * n))
    return torch.tensor(weights, dtype=torch.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Stage 4 ID type classifier")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=_BATCH)
    parser.add_argument("--lr", type=float, default=_LR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip ImageNet weight download",
    )
    args = parser.parse_args()

    if not (DATA_DIR / "train").is_dir():
        print(f"Train split not found: {DATA_DIR / 'train'}", file=sys.stderr)
        print(
            "Run scripts/convert_labels_to_yolo.py then scripts/split_id_type_dataset.py first.",
            file=sys.stderr,
        )
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
    # Normalise device string: "0" → "cuda:0", "cuda" stays, "cpu" stays
    device_str = args.device
    if device_str.isdigit():
        device_str = f"cuda:{device_str}"
    device = torch.device(device_str)

    train_ds = _build_dataset("train", augment=True)
    val_ds = _build_dataset("val", augment=False)

    if not train_ds.samples:
        print("No training images found — check your data split.", file=sys.stderr)
        return 1

    # Print class distribution
    from collections import Counter
    dist = Counter(lbl for _, lbl in train_ds.samples)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    print("Train class distribution:")
    for i, cls in enumerate(CLASSES):
        n = dist.get(i, 0)
        print(f"  {cls:20s}: {n}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = _build_model(pretrained=not args.no_pretrained, num_classes=len(CLASSES))
    model = model.to(device)

    class_weights = _compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        print(f"Epoch {epoch:3d}  loss={total_loss / max(len(train_loader), 1):.4f}  val_acc={val_acc:.4f}")

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
