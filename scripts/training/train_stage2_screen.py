"""Train Stage 2 — Quality Gate classifier (EfficientNet-B0).

8 classes: screen (0), printout (1), selfie (2), back (3), garbage (4),
           good_front (5), partial (6), blurry (7).

Usage:
    python scripts/training/train_stage2_screen.py
    python scripts/training/train_stage2_screen.py --epochs 40 --device cuda

Output: models/stage2_screen/best.pt
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "yolo" / "screen"
OUTPUT_DIR = PROJECT_ROOT / "models" / "stage2_screen"

_IMG_SIZE = 224
_BATCH = 32
_LR = 1e-4
_PATIENCE = 10
_NUM_CLASSES = 8
_CLASS_NAMES = ("screen", "printout", "selfie", "back", "garbage", "good_front", "partial", "blurry")


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
            T.RandomRotation(15),
            # Heavy 90°/180°/270° rotations so the model learns that rotated
            # IDs are still good_front/partial, not garbage.
            T.RandomApply([T.RandomRotation((90, 90))], p=0.25),
            T.RandomApply([T.RandomRotation((180, 180))], p=0.15),
            T.RandomApply([T.RandomRotation((270, 270))], p=0.25),
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
            # If images live on a slow FUSE mount (e.g. Google Drive symlinks),
            # copy them to the local VM disk so DataLoader workers can read fast.
            self._maybe_localise()

        def _maybe_localise(self) -> None:
            """Copy Drive-symlinked images to /tmp so workers read from local NVMe."""
            if not self.samples:
                return
            first = self.samples[0][0].resolve()
            drive_marker = "/content/drive"
            if drive_marker not in str(first):
                return  # already local — nothing to do
            local_dir = Path(tempfile.mkdtemp(prefix="stage2_imgs_"))
            print(f"  Copying {len(self.samples)} images to local disk ({local_dir}) ...")
            new_samples = []
            for img_path, label in self.samples:
                dst = local_dir / img_path.name
                if not dst.exists():
                    shutil.copy2(img_path.resolve(), dst)
                new_samples.append((dst, label))
            self.samples = new_samples
            print(f"  Done — images now on fast local disk.")

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            img_path, label = self.samples[idx]
            img = cv2.imread(str(img_path))
            if img is None:
                # Return a blank tensor on read failure rather than crashing a worker
                import numpy as np
                img = np.zeros((_IMG_SIZE, _IMG_SIZE, 3), dtype=np.uint8)
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

    use_cuda = device.type == "cuda"
    num_workers = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=num_workers, pin_memory=use_cuda, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=num_workers, pin_memory=use_cuda, persistent_workers=num_workers > 0,
    )

    model = _build_model(pretrained=not args.no_pretrained)
    model = model.to(device)

    # Label smoothing reduces overconfidence and helps generalise on small classes
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    patience_counter = 0
    epoch = 0

    def _run_val() -> float:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total else 0.0

    def _train_phase(
        n_epochs: int, lr: float, freeze_backbone: bool, phase_name: str
    ) -> None:
        nonlocal best_val_acc, patience_counter, epoch

        for param in model.features.parameters():
            param.requires_grad = not freeze_backbone

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        print(f"\n[{phase_name}] backbone={'frozen' if freeze_backbone else 'unfrozen'}  lr={lr}  epochs={n_epochs}")

        for _ in range(n_epochs):
            epoch += 1
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

            val_acc = _run_val()
            print(f"Epoch {epoch:3d}  loss={total_loss / len(train_loader):.4f}  val_acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), str(OUTPUT_DIR / "best.pt"))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= _PATIENCE:
                    print(f"Early stop at epoch {epoch} (patience={_PATIENCE})")
                    return

    # Phase 1: train only the classifier head — backbone frozen.
    # Lets the new head converge without destroying pretrained features.
    _train_phase(n_epochs=10, lr=args.lr * 3, freeze_backbone=True, phase_name="Phase 1 — head only")

    # Phase 2: unfreeze everything and fine-tune with a lower LR.
    patience_counter = 0
    _train_phase(n_epochs=args.epochs, lr=args.lr, freeze_backbone=False, phase_name="Phase 2 — full model")

    torch.save(model.state_dict(), str(OUTPUT_DIR / "last.pt"))
    print(f"\nBest val acc: {best_val_acc:.4f}")
    print(f"Weights: {OUTPUT_DIR / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
