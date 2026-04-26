"""
SageMaker entry point for RF-DETR scoreboard detection training.

SageMaker invokes this script inside a PyTorch DLC container on a GPU instance.
It handles the SageMaker-specific directory conventions:

    /opt/ml/input/data/training/    <- S3 dataset downloaded here (COCO format)
        ├── train/
        │   ├── _annotations.coco.json
        │   └── *.jpg
        ├── valid/
        │   ├── _annotations.coco.json
        │   └── *.jpg
        └── test/
            ├── _annotations.coco.json
            └── *.jpg

    /opt/ml/model/                  <- Trained model artifacts uploaded to S3
    /opt/ml/output/                 <- Logs and failure info

Hyperparameters are passed as CLI args by SageMaker from the estimator config.

Usage (launched by SageMaker, not directly):
    python entry_scoreboard_detection.py --epochs 50 --batch_size 8 --model base
"""

import argparse
import os
import shutil
import subprocess
import sys


def install_dependencies():
    """Install rfdetr inside the SageMaker container at runtime."""
    print("Installing rfdetr...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "rfdetr[train,loggers]"]
    )
    print("rfdetr installed successfully.")


def parse_args():
    parser = argparse.ArgumentParser()

    # SageMaker environment paths
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
    )
    parser.add_argument(
        "--num_gpus", type=int, default=int(os.environ.get("SM_NUM_GPUS", 1))
    )

    # Model
    parser.add_argument("--model", type=str, default="base", choices=["base", "large"])

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=560)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


# Augmentation config for scoreboard detection.
# Conservative preset since dataset is small (~161 train images).
# Scoreboards are always upright so no vertical flip or heavy rotation.
AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.15,
        "contrast_limit": 0.15,
        "p": 0.4,
    },
    "HueSaturationValue": {
        "hue_shift_limit": 10,
        "sat_shift_limit": 20,
        "val_shift_limit": 15,
        "p": 0.3,
    },
    "GaussianBlur": {"blur_limit": 3, "p": 0.2},
    "GaussNoise": {"std_range": (0.01, 0.03), "p": 0.2},
}


def main():
    args = parse_args()

    # Install rfdetr (not in base PyTorch DLC)
    install_dependencies()

    from rfdetr import RFDETRBase, RFDETRLarge

    # Validate data channel
    data_dir = args.data_dir
    print(f"Data directory: {data_dir}")
    print(f"Contents: {os.listdir(data_dir)}")

    assert os.path.exists(os.path.join(data_dir, "train", "_annotations.coco.json")), (
        f"Training annotations not found in {data_dir}/train/"
    )
    assert os.path.exists(os.path.join(data_dir, "valid", "_annotations.coco.json")), (
        f"Validation annotations not found in {data_dir}/valid/"
    )

    # Working directory for training outputs (inside container)
    work_dir = "/opt/ml/output/data/training_run"
    os.makedirs(work_dir, exist_ok=True)

    # Initialize model
    print(f"\n{'=' * 60}")
    print("RF-DETR Scoreboard Detection Training")
    print(f"{'=' * 60}")
    print(f"Model:       {args.model}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"LR:          {args.lr}")
    print(f"Image size:  {args.image_size}")
    print(f"Grad accum:  {args.grad_accum_steps}")
    print(f"GPUs:        {args.num_gpus}")
    print(f"Augmentation: {AUG_CONFIG}")
    print(f"{'=' * 60}\n")

    if args.model == "large":
        model = RFDETRLarge()
    else:
        model = RFDETRBase()

    # Train
    model.train(
        dataset_dir=data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_accum_steps=args.grad_accum_steps,
        resolution=args.image_size,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        output_dir=work_dir,
        num_workers=args.num_workers,
        device="cuda",
        aug_config=AUG_CONFIG,
    )

    # Copy model artifacts to SM_MODEL_DIR for S3 upload
    print(f"\nCopying model artifacts to {args.model_dir}...")
    os.makedirs(args.model_dir, exist_ok=True)

    for fname in os.listdir(work_dir):
        src = os.path.join(work_dir, fname)
        dst = os.path.join(args.model_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")
        elif os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"  Copied {fname}/")

    # Ensure the best checkpoint is at a predictable path
    for candidate in ["best_checkpoint.pt", "best.pt", "checkpoint.pt"]:
        src = os.path.join(work_dir, candidate)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.model_dir, "model.pt"))
            print("  Best model saved as model.pt")
            break

    print("\nTraining complete! Model artifacts will be uploaded to S3.")


if __name__ == "__main__":
    main()
