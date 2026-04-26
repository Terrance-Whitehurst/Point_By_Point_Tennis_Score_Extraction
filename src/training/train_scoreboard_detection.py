"""
Train RF-DETR model for scoreboard detection.

RF-DETR (Roboflow Detection Transformer) is a real-time object detection model
that natively accepts COCO-format datasets. Our Scoreboard_Detection dataset has
1 class: scoreboards (id=0/1).

Dataset structure expected:
    data/raw/Scoreboard_Detection/
    ├── train/
    │   ├── _annotations.coco.json
    │   └── *.jpg  (161 images)
    ├── valid/
    │   ├── _annotations.coco.json
    │   └── *.jpg
    └── test/
        ├── _annotations.coco.json
        └── *.jpg

Usage:
    python -m src.training.train_scoreboard_detection
    python -m src.training.train_scoreboard_detection --model large --epochs 100 --batch_size 4
    python -m src.training.train_scoreboard_detection --resume models/scoreboard_detection/checkpoint.pt
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rfdetr import RFDETRBase, RFDETRLarge


def get_args():
    parser = argparse.ArgumentParser(
        description="Train RF-DETR for scoreboard detection"
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["base", "large"],
        help="RF-DETR model size (base: faster training, large: better accuracy)",
    )

    # Data
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data/raw/Scoreboard_Detection",
        help="Path to COCO-format dataset directory",
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=560,
        help="Input image resolution (must be multiple of 56)",
    )

    # Optimization
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=0, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step"],
        help="LR scheduler type",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/scoreboard_detection",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume training"
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda, mps, cpu). Auto-detected if not set.",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    return parser.parse_args()


def train(args):
    # Validate dataset exists
    dataset_dir = Path(args.dataset_dir)
    assert dataset_dir.exists(), f"Dataset directory not found: {dataset_dir}"
    assert (dataset_dir / "train" / "_annotations.coco.json").exists(), (
        f"Training annotations not found at {dataset_dir / 'train' / '_annotations.coco.json'}"
    )
    assert (dataset_dir / "valid" / "_annotations.coco.json").exists(), (
        f"Validation annotations not found at {dataset_dir / 'valid' / '_annotations.coco.json'}"
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model
    print(f"Initializing RF-DETR ({args.model})...")
    if args.model == "large":
        model = RFDETRLarge()
    else:
        model = RFDETRBase()

    # Configure and start training
    print(f"Dataset: {args.dataset_dir}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Image size: {args.image_size}, Grad accum: {args.grad_accum_steps}")
    print(f"Output: {args.output_dir}")

    train_kwargs = dict(
        dataset_dir=str(dataset_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_accum_steps=args.grad_accum_steps,
        resolution=args.image_size,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
    )

    if args.resume:
        train_kwargs["checkpoint"] = args.resume

    if args.device:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)

    # Copy best checkpoint to models/ for easy access
    best_ckpt = Path(args.output_dir) / "best_checkpoint.pt"
    if best_ckpt.exists():
        dest = Path("models") / "scoreboard_detection_rfdetr_best.pt"
        os.makedirs(dest.parent, exist_ok=True)
        import shutil

        shutil.copy2(best_ckpt, dest)
        print(f"Best checkpoint copied to {dest}")

    print("Training complete!")


if __name__ == "__main__":
    args = get_args()
    train(args)
