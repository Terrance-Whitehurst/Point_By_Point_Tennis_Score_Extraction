# Point-by-Point Tennis Score Extraction

Two-stage pipeline for extracting point-by-point score data from tennis match videos:

- **Stage 1 — Scoreboard Detection**: RF-DETR fine-tuned to localize the scoreboard overlay on every frame, paired with SSIM-based change detection so OCR only fires when the score visibly changes.
- **Stage 2 — Score OCR**: a vision-language reader (FastVLM via `mlx-vlm` / `transformers`) or a classical OCR reader (EasyOCR with spatial parsing) extracts player names, sets, games, points, and server.

## Project Structure

```
Point_By_Point_Tennis_Score_Extraction/
├── Makefile                  # Convenience commands
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── configs/
│   └── broadcaster_offsets.json  # Per-broadcaster lag/start calibration
│
├── data/                     # Datasets (NOT committed)
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/                   # Trained checkpoints (NOT committed)
│   └── scoreboard_detection/
│
├── reports/                  # Pipeline output CSVs and figures
│
├── src/
│   ├── inference/
│   │   └── scoreboard_ocr.py             # End-to-end pipeline (CLI + library)
│   └── training/
│       └── train_scoreboard_detection.py # RF-DETR fine-tuning
│
├── scripts/
│   └── sagemaker/
│       ├── entry_scoreboard_detection.py     # SageMaker container entrypoint
│       └── launch_scoreboard_detection.py    # SageMaker job launcher
│
└── tests/
```

## Setup

```bash
uv pip install -e .
# or
uv pip install -r requirements.txt
```

`mlx-vlm` (Apple Silicon FastVLM backend) is optional — install separately if you want to use it:
```bash
uv pip install mlx-vlm
```

## Usage

### Run the OCR pipeline on a video

```bash
python -m src.inference.scoreboard_ocr --video path/to/match.mp4
```

Output is a per-point CSV with columns:
```
frame_num, timestamp_sec, player1_name, player2_name,
set_score_p1, set_score_p2, game_score_p1, game_score_p2,
point_score_p1, point_score_p2, server, returner
```

### Programmatic use

```python
from src.inference.scoreboard_ocr import ScoreboardOCRPipeline

pipeline = ScoreboardOCRPipeline(...)
results = pipeline.process_video("match.mp4", "scores.csv")
```

### Train scoreboard detection (RF-DETR)

```bash
# Base model
python -m src.training.train_scoreboard_detection

# Large model with more epochs
python -m src.training.train_scoreboard_detection \
    --model large \
    --epochs 100 \
    --batch_size 4

# Resume from checkpoint
python -m src.training.train_scoreboard_detection \
    --resume models/scoreboard_detection/checkpoint.pt
```

Dataset: `data/raw/Scoreboard_Detection/` (COCO format). Class: `scoreboard`.

### Train on AWS SageMaker

```bash
ROLE_ARN="arn:aws:iam::<ACCOUNT_ID>:role/<SAGEMAKER_ROLE>"

# Basic — uploads local data, trains on ml.g4dn.xlarge
python scripts/sagemaker/launch_scoreboard_detection.py --role $ROLE_ARN

# Larger instance + model
python scripts/sagemaker/launch_scoreboard_detection.py \
    --role $ROLE_ARN \
    --instance_type ml.g5.xlarge \
    --model large \
    --epochs 100 \
    --batch_size 16

# Data already in S3
python scripts/sagemaker/launch_scoreboard_detection.py \
    --role $ROLE_ARN \
    --s3_data s3://my-bucket/datasets/Scoreboard_Detection
```

### Retrieving trained model

```bash
aws s3 cp s3://training-jobs-test-315109499400/tennis-analysis/models/scoreboard_detection/<job>/output/model.tar.gz \
    models/scoreboard_detection/
tar xzf models/scoreboard_detection/model.tar.gz -C models/scoreboard_detection/
```

## How It Works

1. **Per-frame detection (cheap):** RF-DETR localizes the scoreboard. The detected crop is resampled to a canonical size and compared with the previous accepted crop using structural similarity (SSIM).
2. **Change-gated OCR (expensive):** When SSIM drops below a threshold, a small state machine waits for the new scoreboard to settle, then sends a single crop to the OCR backend. Stable readings are written as one row per scoring event.

This is roughly 10–100× cheaper than running OCR every frame, while still capturing every score change.

## References

- [RF-DETR](https://github.com/roboflow/rf-detr) — Real-time Detection Transformer
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [FastVLM](https://huggingface.co/apple/FastVLM-1.5B) — Apple's small vision-language model
