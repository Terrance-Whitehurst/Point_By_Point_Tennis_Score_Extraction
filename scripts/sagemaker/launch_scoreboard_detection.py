"""
Launch RF-DETR scoreboard detection training on AWS SageMaker.

Uploads the dataset to S3 (if needed), configures a PyTorch training job,
and submits it to SageMaker. The trained model is saved back to S3.

Prerequisites:
    1. AWS CLI configured (`aws configure`)
    2. SageMaker execution role with S3 access
    3. Dataset at data/raw/Scoreboard_Detection/ (or already on S3)
    4. pip install sagemaker boto3

Usage:
    # Basic launch (uploads data, starts training)
    python scripts/sagemaker/launch_scoreboard_detection.py \
        --role arn:aws:iam::123456789012:role/SageMakerRole

    # Use data already on S3
    python scripts/sagemaker/launch_scoreboard_detection.py \
        --role arn:aws:iam::123456789012:role/SageMakerRole \
        --s3_data s3://my-bucket/datasets/Scoreboard_Detection

    # Custom instance and hyperparams
    python scripts/sagemaker/launch_scoreboard_detection.py \
        --role arn:aws:iam::123456789012:role/SageMakerRole \
        --instance_type ml.g5.xlarge \
        --epochs 100 \
        --batch_size 16 \
        --model large
"""

import argparse
import os
from datetime import datetime

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch RF-DETR scoreboard training on SageMaker"
    )

    # AWS / SageMaker
    parser.add_argument(
        "--role", type=str, required=True, help="SageMaker execution role ARN"
    )
    parser.add_argument(
        "--region", type=str, default=None, help="AWS region (auto-detected if not set)"
    )
    parser.add_argument(
        "--instance_type",
        type=str,
        default="ml.g4dn.xlarge",
        help="SageMaker instance type (ml.g4dn.xlarge, ml.g5.xlarge, ml.g5.2xlarge, ml.p3.2xlarge)",
    )
    parser.add_argument(
        "--instance_count", type=int, default=1, help="Number of training instances"
    )
    parser.add_argument(
        "--volume_size", type=int, default=50, help="EBS volume size in GB"
    )
    parser.add_argument(
        "--max_runtime",
        type=int,
        default=86400,
        help="Max training time in seconds (default: 24h)",
    )
    parser.add_argument(
        "--spot",
        action="store_true",
        help="Use spot instances (cheaper but can be interrupted)",
    )

    # Data
    parser.add_argument(
        "--s3_data",
        type=str,
        default=None,
        help="S3 URI of dataset. If not set, uploads from --local_data",
    )
    parser.add_argument(
        "--local_data",
        type=str,
        default="data/raw/Scoreboard_Detection",
        help="Local dataset path to upload to S3",
    )
    parser.add_argument(
        "--s3_bucket",
        type=str,
        default=None,
        help="S3 bucket for data upload (default: SageMaker default bucket)",
    )
    parser.add_argument(
        "--s3_prefix",
        type=str,
        default="tennis-analysis/datasets/Scoreboard_Detection",
        help="S3 prefix for data upload",
    )

    # Model output
    parser.add_argument(
        "--s3_output",
        type=str,
        default=None,
        help="S3 URI for model output (default: s3://<bucket>/tennis-analysis/models/)",
    )

    # Training hyperparameters (passed to entry point)
    parser.add_argument("--model", type=str, default="base", choices=["base", "large"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=560)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    # Job config
    parser.add_argument(
        "--job_name",
        type=str,
        default=None,
        help="Training job name (auto-generated if not set)",
    )
    parser.add_argument(
        "--wait", action="store_true", help="Wait for job to complete (blocks)"
    )
    parser.add_argument("--tags", type=str, nargs="*", help="Tags as key=value pairs")

    return parser.parse_args()


def upload_data_to_s3(local_path, bucket, prefix, session):
    """Upload local dataset to S3 and return the S3 URI."""
    s3_uri = f"s3://{bucket}/{prefix}"
    print(f"Uploading {local_path} -> {s3_uri}")
    s3_uri = sagemaker.Session(boto_session=session).upload_data(
        path=local_path,
        bucket=bucket,
        key_prefix=prefix,
    )
    print(f"Upload complete: {s3_uri}")
    return s3_uri


def main():
    args = parse_args()

    # Initialize SageMaker session
    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)
    bucket = args.s3_bucket or sm_session.default_bucket()

    # Upload data to S3 if needed
    if args.s3_data:
        s3_data_uri = args.s3_data
        print(f"Using existing S3 data: {s3_data_uri}")
    else:
        assert os.path.exists(args.local_data), (
            f"Local data not found: {args.local_data}"
        )
        s3_data_uri = upload_data_to_s3(
            args.local_data, bucket, args.s3_prefix, boto_session
        )

    # S3 output path
    s3_output = (
        args.s3_output or f"s3://{bucket}/tennis-analysis/models/scoreboard_detection"
    )

    # Job name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = args.job_name or f"rfdetr-scoreboard-{args.model}-{timestamp}"

    # Tags
    tags = []
    if args.tags:
        for tag in args.tags:
            k, v = tag.split("=", 1)
            tags.append({"Key": k, "Value": v})
    tags.append({"Key": "project", "Value": "tennis-analysis"})
    tags.append({"Key": "task", "Value": "scoreboard-detection"})

    # Spot instance config
    checkpoint_s3 = (
        f"s3://{bucket}/tennis-analysis/checkpoints/scoreboard_detection/"
        if args.spot
        else None
    )

    print(f"\n{'=' * 60}")
    print(f"SageMaker Training Job: {job_name}")
    print(f"{'=' * 60}")
    print(f"Instance:    {args.instance_type} x {args.instance_count}")
    print(f"Spot:        {args.spot}")
    print(f"Data:        {s3_data_uri}")
    print(f"Output:      {s3_output}")
    print(f"Model:       RF-DETR {args.model}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"LR:          {args.lr}")
    print(f"Image size:  {args.image_size}")
    print(f"{'=' * 60}\n")

    # Configure PyTorch estimator
    estimator = PyTorch(
        entry_point="entry_scoreboard_detection.py",
        source_dir="scripts/sagemaker",
        role=args.role,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        framework_version="2.5.1",
        py_version="py311",
        output_path=s3_output,
        volume_size=args.volume_size,
        max_run=args.max_runtime,
        use_spot_instances=args.spot,
        max_wait=args.max_runtime * 2 if args.spot else None,
        checkpoint_s3_uri=checkpoint_s3,
        sagemaker_session=sm_session,
        tags=tags,
        hyperparameters={
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "grad_accum_steps": args.grad_accum_steps,
            "image_size": args.image_size,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "num_workers": args.num_workers,
        },
    )

    # Launch training
    estimator.fit(
        inputs={"training": s3_data_uri},
        job_name=job_name,
        wait=args.wait,
        logs="All" if args.wait else None,
    )

    if args.wait:
        print("\nTraining complete!")
        print(f"Model artifacts: {estimator.model_data}")
    else:
        print(f"\nTraining job submitted: {job_name}")
        print(
            f"Monitor at: https://{boto_session.region_name}.console.aws.amazon.com/sagemaker/home#/jobs/{job_name}"
        )

    return estimator


if __name__ == "__main__":
    main()
