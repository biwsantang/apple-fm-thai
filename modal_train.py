"""Modal deployment for Apple FM Thai adapter training.

Setup (one-time):
    pip install modal
    modal setup
    modal volume create apple-fm-toolkit
    modal volume create apple-fm-thai
    modal volume put apple-fm-toolkit /path/to/apple-fm-toolkit/ /toolkit/

Upload data (per iteration, named by commit hash):
    HASH=$(git rev-parse --short HEAD)
    modal volume put apple-fm-thai /path/to/data/processed/iteration_1/ /data/$HASH/

Run training (auto-detects current commit hash):
    modal run modal_train.py
    modal run modal_train.py --epochs 1              # quick test
    modal run modal_train.py --run-id abc1234        # specific commit hash

Download checkpoints:
    modal volume get apple-fm-thai /checkpoints/$HASH/ ./adapter/$HASH/

Volumes:
    apple-fm-toolkit  — shared base model + tokenizer (~12GB, reuse across experiments)
    apple-fm-thai     — experiment data + checkpoints (keyed by commit hash)
"""

import modal
import subprocess

app = modal.App("apple-fm-thai")

# Shared: base model + tokenizer (reuse across all experiments)
toolkit_volume = modal.Volume.from_name("apple-fm-toolkit")

# Experiment: data + checkpoints (keyed by commit hash)
experiment_volume = modal.Volume.from_name("apple-fm-thai")

# Flash Attention prebuilt wheel (must match torch + CUDA + Python version exactly)
FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
)

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6,<2.7",  # pinned for flash-attn wheel compatibility
        "tamm~=0.1.0",
        "sentencepiece>=0.2.1",
        "pydantic>=2.12.5",
        "tqdm>=4.67.3",
        "rich>=14.3.3",
        FLASH_ATTN_WHEEL,
        gpu="a100",
    )
)

TOOLKIT_MOUNT = "/toolkit"
EXPERIMENT_MOUNT = "/experiment"


def get_commit_hash() -> str:
    """Get short commit hash from local git repo."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    )
    return result.stdout.strip()


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={
        TOOLKIT_MOUNT: toolkit_volume,
        EXPERIMENT_MOUNT: experiment_volume,
    },
    timeout=3600,
)
def train(
    run_id: str,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
):
    import os
    import torch

    # CUDA optimizations
    torch.backends.cudnn.benchmark = True          # auto-tune CUDA kernels
    torch.backends.cuda.matmul.allow_tf32 = True   # ~3x faster matmul on A100
    torch.backends.cudnn.allow_tf32 = True         # TF32 for cuDNN ops

    # Locate toolkit (shared) and data (per run)
    toolkit_dir = f"{TOOLKIT_MOUNT}/toolkit"
    train_data = f"{EXPERIMENT_MOUNT}/data/{run_id}/train.jsonl"
    eval_data = f"{EXPERIMENT_MOUNT}/data/{run_id}/eval.jsonl"
    checkpoint_dir = f"{EXPERIMENT_MOUNT}/checkpoints/{run_id}"

    # Verify assets exist
    assert os.path.exists(os.path.join(toolkit_dir, "assets", "base-model.pt")), \
        "Toolkit not found. Upload with: modal volume put apple-fm-toolkit /path/to/apple-fm-toolkit/ /toolkit/"
    assert os.path.exists(train_data), \
        f"Data not found for {run_id}. Upload with: modal volume put apple-fm-thai /path/to/data/ /data/{run_id}/"

    print(f"=== Run: {run_id} ===")
    print(f"    Toolkit: {toolkit_dir}")
    print(f"    Train: {train_data}")
    print(f"    Eval: {eval_data}")
    print(f"    Checkpoints: {checkpoint_dir}")

    # Train
    print("\n=== Starting training ===")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{toolkit_dir}:{env.get('PYTHONPATH', '')}"
    env["PYTHONHASHSEED"] = "42"

    os.makedirs(checkpoint_dir, exist_ok=True)

    cmd = [
        "python", "-m", "examples.train_adapter",
        "--train-data", train_data,
        "--eval-data", eval_data,
        "--epochs", str(epochs),
        "--learning-rate", str(learning_rate),
        "--batch-size", str(batch_size),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--pack-sequences",
        "--max-sequence-length", "4095",
        "--activation-checkpointing",
        "--precision", "bf16-mixed",
        "--checkpoint-frequency", "1",
        "--weight-decay", "0.01",
        "--clip-grad-norm", "1.0",
        "--compile-model",
        "--checkpoint-dir", checkpoint_dir,
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    # List checkpoints
    if os.path.exists(checkpoint_dir):
        files = os.listdir(checkpoint_dir)
        print(f"\n=== Checkpoints ({len(files)} files) ===")
        for f in sorted(files):
            size = os.path.getsize(os.path.join(checkpoint_dir, f))
            print(f"  {f}: {size / 1024 / 1024:.1f} MB")

    experiment_volume.commit()
    print(f"\n=== Done! ===")
    print(f"Download: modal volume get apple-fm-thai /checkpoints/{run_id}/ ./adapter/{run_id}/")

    return "Training complete"


@app.local_entrypoint()
def main(
    run_id: str = "",
    epochs: int = 3,
    learning_rate: float = 1e-3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
):
    if not run_id:
        run_id = get_commit_hash()

    print(f"Launching run {run_id}: epochs={epochs}, lr={learning_rate}, bs={batch_size}")
    result = train.remote(
        run_id=run_id,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    print(result)
    print(f"\nTo download checkpoints:")
    print(f"  modal volume get apple-fm-thai /checkpoints/{run_id}/ ./adapter/{run_id}/")
