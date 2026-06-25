"""Benchmark DINOv2 vs EfficientNet-B0 on the Stage 2 quality-gate dataset.

Three approaches are compared head-to-head on the same val split:

  1. EfficientNet-B0 fine-tuned     (existing model — evaluated as-is)
  2. DINOv2-ViT-S/14 linear probe   (frozen backbone + 1 linear layer, fast)
  3. DINOv2-ViT-S/14 + MLP head     (frozen backbone + 2-layer MLP, slightly more capacity)

DINOv2 models 2 & 3 are trained here from scratch (head only, backbone frozen).
Training is fast — typically 10–20 epochs to converge on frozen features.

Usage:
    python scripts/training/benchmark_stage2_dinov2.py --device cuda
    python scripts/training/benchmark_stage2_dinov2.py --device cuda --epochs 30 --also-finetune

Output:
    - Side-by-side macro-F1 / accuracy / per-class table printed to stdout
    - DINOv2 linear probe weights saved to models/stage2_dinov2_linear/best.pt
    - DINOv2 MLP head weights saved to models/stage2_dinov2_mlp/best.pt
    - (if --also-finetune) Full fine-tune saved to models/stage2_dinov2_finetune/best.pt
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "yolo" / "screen"
EFFICIENTNET_WEIGHTS = PROJECT_ROOT / "models" / "stage2_screen" / "best.pt"

_IMG_SIZE = 224
_BATCH = 64   # larger batch is fine for frozen backbone (no gradient through it)
_LR_HEAD = 1e-3
_LR_FINETUNE = 5e-5
_PATIENCE = 8
_NUM_CLASSES = 8
_CLASS_NAMES = ("screen", "printout", "selfie", "back", "garbage", "good_front", "partial", "blurry")

# DINOv2 models — ordered from fastest to most accurate
_DINOV2_VARIANTS = {
    "vits14": ("facebookresearch/dinov2", "dinov2_vits14", 384),   # 21M params
    "vitb14": ("facebookresearch/dinov2", "dinov2_vitb14", 768),   # 86M params
}


# ── Data loading ─────────────────────────────────────────────────────────────

def _build_dataset(split: str, transform):
    """Build dataset using the same image/label directory structure as train_stage2_screen.py."""
    import cv2  # type: ignore
    import torch  # type: ignore
    from torch.utils.data import Dataset  # type: ignore

    img_dir = DATA_DIR / "images" / split
    lbl_dir = DATA_DIR / "labels" / split
    files = sorted(img_dir.glob("*.jpg"))

    samples: list[tuple[Path, int]] = []
    for img_path in files:
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if lbl.is_file():
            samples.append((img_path, int(lbl.read_text().strip())))

    # Copy Drive-symlinked images to local /tmp for fast I/O
    if samples:
        first = samples[0][0].resolve()
        if "/content/drive" in str(first):
            local_dir = Path(tempfile.mkdtemp(prefix=f"dinov2_{split}_"))
            print(f"  [{split}] Copying {len(samples)} images to local disk ...")
            new = []
            for p, lbl in samples:
                dst = local_dir / p.name
                if not dst.exists():
                    shutil.copy2(p.resolve(), dst)
                new.append((dst, lbl))
            samples = new
            print(f"  [{split}] Done.")

    class _DS(Dataset):
        def __init__(self) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            path, label = self.samples[idx]
            img = cv2.imread(str(path))
            if img is None:
                import numpy as np
                img = np.zeros((_IMG_SIZE, _IMG_SIZE, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return transform(img), label

    return _DS()


def _make_transforms(augment: bool):
    import torchvision.transforms as T  # type: ignore

    base = [T.ToPILImage(), T.Resize((_IMG_SIZE, _IMG_SIZE))]
    aug = [
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        T.RandomRotation(15),
        T.RandomApply([T.RandomRotation((90, 90))], p=0.25),
        T.RandomApply([T.RandomRotation((180, 180))], p=0.15),
        T.RandomApply([T.RandomRotation((270, 270))], p=0.25),
    ] if augment else []
    tail = [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return T.Compose(base + aug + tail)


def _class_weights(samples, device):
    import torch  # type: ignore

    counts = Counter(lbl for _, lbl in samples)
    total = len(samples)
    weights = [total / (_NUM_CLASSES * counts.get(c, 1)) for c in range(_NUM_CLASSES)]
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ── Metrics ──────────────────────────────────────────────────────────────────

def _compute_metrics(model, loader, device) -> dict:
    import torch  # type: ignore

    model.eval()
    correct = total = 0
    tp: dict[int, int] = {}
    pred_pos: dict[int, int] = {}
    real_pos: dict[int, int] = {}

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for gt, pr in zip(labels.tolist(), preds.tolist()):
                tp[gt] = tp.get(gt, 0) + (1 if gt == pr else 0)
                pred_pos[pr] = pred_pos.get(pr, 0) + 1
                real_pos[gt] = real_pos.get(gt, 0) + 1

    f1s = []
    per_class = {}
    for c in range(_NUM_CLASSES):
        t = tp.get(c, 0)
        pp = pred_pos.get(c, 0)
        rp = real_pos.get(c, 0)
        prec = t / pp if pp else 0.0
        rec = t / rp if rp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[_CLASS_NAMES[c]] = {"precision": round(prec, 3), "recall": round(rec, 3),
                                      "f1": round(f1, 3), "support": rp}
        if rp > 0:
            f1s.append(f1)

    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "per_class": per_class,
    }


# ── EfficientNet evaluation ───────────────────────────────────────────────────

def evaluate_efficientnet(val_loader, device) -> dict | None:
    if not EFFICIENTNET_WEIGHTS.is_file():
        print(f"  EfficientNet weights not found at {EFFICIENTNET_WEIGHTS} — skipping.")
        return None

    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torchvision.models import efficientnet_b0  # type: ignore

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, _NUM_CLASSES)
    state = torch.load(str(EFFICIENTNET_WEIGHTS), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    print("  Evaluating EfficientNet-B0 (existing weights)...")
    return _compute_metrics(model, val_loader, device)


# ── DINOv2 head training ──────────────────────────────────────────────────────

def _load_dinov2(variant: str, device):
    repo, name, feat_dim = _DINOV2_VARIANTS[variant]
    print(f"  Loading {name} from torch.hub ...")
    backbone = __import__("torch").hub.load(repo, name, pretrained=True)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone.to(device), feat_dim


class _LinearHead(__import__("torch").nn.Module):
    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        import torch.nn as nn  # type: ignore
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class _MLPHead(__import__("torch").nn.Module):
    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        import torch.nn as nn  # type: ignore
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class _DinoWrapper(__import__("torch").nn.Module):
    """Backbone + head wrapped as a single model (needed for _compute_metrics)."""

    def __init__(self, backbone, head) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        with __import__("torch").no_grad():
            feats = self.backbone(x)   # [B, feat_dim] — DINOv2 returns CLS token
        return self.head(feats)


def train_dinov2_head(
    variant: str,
    head_type: str,   # "linear" | "mlp"
    train_loader,
    val_loader,
    device,
    epochs: int,
    class_weights,
    output_dir: Path,
) -> dict:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    backbone, feat_dim = _load_dinov2(variant, device)

    head: nn.Module
    if head_type == "linear":
        head = _LinearHead(feat_dim, _NUM_CLASSES).to(device)
    else:
        head = _MLPHead(feat_dim, _NUM_CLASSES).to(device)

    wrapper = _DinoWrapper(backbone, head)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = __import__("torch").optim.AdamW(head.parameters(), lr=_LR_HEAD, weight_decay=1e-4)
    scheduler = __import__("torch").optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    patience_counter = 0

    print(f"\n  Training DINOv2-{variant} {head_type} head ({epochs} epochs) ...")
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feats = backbone(imgs)
            optimizer.zero_grad()
            loss = criterion(head(feats), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        metrics = _compute_metrics(wrapper, val_loader, device)
        mf1 = metrics["macro_f1"]
        print(f"  Epoch {epoch:3d}  loss={total_loss / len(train_loader):.4f}"
              f"  val_acc={metrics['accuracy']:.4f}  macro_f1={mf1:.4f}")

        if mf1 > best_f1:
            best_f1 = mf1
            torch.save(head.state_dict(), str(output_dir / "best_head.pt"))
            torch.save({"variant": variant, "head_type": head_type, "feat_dim": feat_dim},
                       str(output_dir / "config.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= _PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    # Reload best head for final eval
    head.load_state_dict(torch.load(str(output_dir / "best_head.pt"), weights_only=True))
    return _compute_metrics(wrapper, val_loader, device)


def train_dinov2_finetune(
    variant: str,
    train_loader,
    val_loader,
    device,
    epochs: int,
    class_weights,
    output_dir: Path,
) -> dict:
    """Full fine-tune — unfreezes the entire DINOv2 backbone (slow, requires A100-class GPU)."""
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    backbone, feat_dim = _load_dinov2(variant, device)
    for p in backbone.parameters():
        p.requires_grad_(True)

    head = _MLPHead(feat_dim, _NUM_CLASSES).to(device)
    wrapper = _DinoWrapper(backbone, head)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW([
        {"params": backbone.parameters(), "lr": _LR_FINETUNE},
        {"params": head.parameters(),     "lr": _LR_HEAD},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = 0.0
    patience_counter = 0

    print(f"\n  Full fine-tune DINOv2-{variant} ({epochs} epochs) ...")
    for epoch in range(1, epochs + 1):
        wrapper.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(wrapper(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        metrics = _compute_metrics(wrapper, val_loader, device)
        mf1 = metrics["macro_f1"]
        print(f"  Epoch {epoch:3d}  loss={total_loss / len(train_loader):.4f}"
              f"  val_acc={metrics['accuracy']:.4f}  macro_f1={mf1:.4f}")

        if mf1 > best_f1:
            best_f1 = mf1
            torch.save(wrapper.state_dict(), str(output_dir / "best.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= _PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    wrapper.load_state_dict(torch.load(str(output_dir / "best.pt"), weights_only=True))
    return _compute_metrics(wrapper, val_loader, device)


# ── Report ────────────────────────────────────────────────────────────────────

def _print_report(results: dict[str, dict]) -> None:
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS — Stage 2 Quality Gate")
    print("=" * 80)

    header = f"{'Model':<35} {'Accuracy':>10} {'Macro-F1':>10}"
    print(header)
    print("-" * 55)
    for name, m in results.items():
        print(f"{name:<35} {m['accuracy']:>10.4f} {m['macro_f1']:>10.4f}")

    print("\nPer-class F1:")
    cls_header = f"  {'Class':<15}" + "".join(f"{n[:18]:>20}" for n in results)
    print(cls_header)
    print("  " + "-" * (15 + 20 * len(results)))
    for c in _CLASS_NAMES:
        row = f"  {c:<15}"
        for m in results.values():
            pc = m["per_class"].get(c, {})
            f1 = pc.get("f1", 0.0)
            sup = pc.get("support", 0)
            row += f"{f1:>14.3f} ({sup:3d})"
        print(row)

    print("\nKey: (n) = val support for that class")
    print("=" * 80)

    # Safety check: fraud-as-legit analysis
    print("\nFRAUD SLIP-THROUGH ANALYSIS (reject class predicted as accept class):")
    print("  Reject classes: screen, printout, selfie, back, garbage")
    print("  Accept classes: good_front, partial, blurry")
    print("  (Lower recall on reject classes = more fraud slipping through)")
    print()
    reject_classes = {"screen", "printout", "selfie", "back", "garbage"}
    for name, m in results.items():
        missed = []
        for c in reject_classes:
            pc = m["per_class"].get(c, {})
            rec = pc.get("recall", 1.0)
            sup = pc.get("support", 0)
            missed_n = round((1 - rec) * sup)
            if missed_n > 0:
                missed.append(f"{c}:{missed_n}/{sup}")
        print(f"  {name:<35} fraud missed: {', '.join(missed) if missed else 'none'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark DINOv2 vs EfficientNet-B0 on Stage 2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=25,
                        help="Epochs for DINOv2 head training (linear probe converges fast)")
    parser.add_argument("--variant", choices=list(_DINOV2_VARIANTS), default="vits14",
                        help="DINOv2 variant: vits14 (21M, fast) or vitb14 (86M, better)")
    parser.add_argument("--also-finetune", action="store_true",
                        help="Also run full DINOv2 fine-tune (slow — needs A100-class GPU)")
    parser.add_argument("--skip-efficientnet", action="store_true",
                        help="Skip EfficientNet evaluation (if no weights exist yet)")
    parser.add_argument("--batch", type=int, default=_BATCH)
    args = parser.parse_args()

    raw_dev = args.device.strip()
    if raw_dev.isdigit():
        raw_dev = f"cuda:{raw_dev}"

    if not (DATA_DIR / "images" / "train").is_dir():
        print(f"Stage 2 dataset not found at {DATA_DIR}", file=sys.stderr)
        print("Run Section 3 in the notebook first.", file=sys.stderr)
        return 1

    try:
        import torch  # type: ignore
        from torch.utils.data import DataLoader  # type: ignore
    except ImportError as exc:
        print(f"Missing: {exc}", file=sys.stderr)
        return 1

    device = torch.device(raw_dev)
    num_workers = min(4, os.cpu_count() or 2)

    train_tf = _make_transforms(augment=True)
    val_tf = _make_transforms(augment=False)

    print("Loading datasets ...")
    train_ds = _build_dataset("train", train_tf)
    val_ds = _build_dataset("val", val_tf)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"),
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=num_workers, pin_memory=(device.type == "cuda"),
                            persistent_workers=num_workers > 0)

    weights = _class_weights(train_ds.samples, device)

    results: dict[str, dict] = {}

    # 1. EfficientNet-B0 (existing weights)
    if not args.skip_efficientnet:
        print("\n[1/3] EfficientNet-B0 (existing trained weights)")
        m = evaluate_efficientnet(val_loader, device)
        if m:
            results["EfficientNet-B0 (fine-tuned)"] = m

    # 2. DINOv2 linear probe
    print(f"\n[2/3] DINOv2-{args.variant} linear probe (frozen backbone)")
    out_linear = PROJECT_ROOT / "models" / f"stage2_dinov2_{args.variant}_linear"
    m = train_dinov2_head(args.variant, "linear", train_loader, val_loader,
                          device, args.epochs, weights, out_linear)
    results[f"DINOv2-{args.variant} linear probe"] = m

    # 3. DINOv2 MLP head
    print(f"\n[3/3] DINOv2-{args.variant} MLP head (frozen backbone)")
    out_mlp = PROJECT_ROOT / "models" / f"stage2_dinov2_{args.variant}_mlp"
    m = train_dinov2_head(args.variant, "mlp", train_loader, val_loader,
                          device, args.epochs, weights, out_mlp)
    results[f"DINOv2-{args.variant} MLP head"] = m

    # 4. Optional full fine-tune
    if args.also_finetune:
        print(f"\n[4/4] DINOv2-{args.variant} full fine-tune (slow!)")
        out_ft = PROJECT_ROOT / "models" / f"stage2_dinov2_{args.variant}_finetune"
        m = train_dinov2_finetune(args.variant, train_loader, val_loader,
                                  device, args.epochs, weights, out_ft)
        results[f"DINOv2-{args.variant} fine-tuned"] = m

    _print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
