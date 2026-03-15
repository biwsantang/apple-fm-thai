"""Modal deployment for Apple FM Thai adapter training.

Setup (one-time):
    pip install modal
    modal setup
    modal volume create apple-fm-toolkit
    modal volume put apple-fm-toolkit /path/to/apple-fm-toolkit/ /toolkit/
    modal volume put apple-fm-toolkit /path/to/data/processed/iteration_1/ /data/

Run training:
    modal run modal_train.py
    modal run modal_train.py --epochs 1    # quick test
"""

import modal

app = modal.App("apple-fm-thai")

# Persistent volume for toolkit assets + processed data + checkpoints
toolkit_volume = modal.Volume.from_name("apple-fm-toolkit")

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6",
        "tamm~=0.1.0",
        "sentencepiece>=0.2.1",
        "pydantic>=2.12.5",
        "tqdm>=4.67.3",
        "rich>=14.3.3",
        gpu="a100",
    )
)

VOL_MOUNT = "/vol"


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOL_MOUNT: toolkit_volume},
    timeout=3600,
)
def train(
    epochs: int = 3,
    learning_rate: float = 1e-3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
):
    import subprocess
    import os
    import shutil

    # Locate toolkit and data on the volume
    toolkit_dir = f"{VOL_MOUNT}/toolkit"
    train_data = f"{VOL_MOUNT}/data/train.jsonl"
    eval_data = f"{VOL_MOUNT}/data/eval.jsonl"
    checkpoint_dir = f"{VOL_MOUNT}/checkpoints/iteration_1"

    # Verify assets exist
    assert os.path.exists(os.path.join(toolkit_dir, "assets", "base-model.pt")), \
        f"Toolkit not found at {toolkit_dir}. Upload with: modal volume put apple-fm-toolkit /path/to/apple-fm-toolkit/ /toolkit/"
    assert os.path.exists(train_data), \
        f"Training data not found at {train_data}. Upload with: modal volume put apple-fm-toolkit /path/to/data/processed/iteration_1/ /data/"

    print(f"=== Toolkit: {os.listdir(os.path.join(toolkit_dir, 'assets'))} ===")
    print(f"=== Train data: {train_data} ===")
    print(f"=== Eval data: {eval_data} ===")

    # Train
    print("=== Starting training ===")
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

    toolkit_volume.commit()
    print(f"\n=== Done! Checkpoints saved to volume at /checkpoints/iteration_1/ ===")
    print("Download with: modal volume get apple-fm-toolkit /checkpoints/ ./adapter/")

    return "Training complete"


@app.local_entrypoint()
def main(
    epochs: int = 3,
    learning_rate: float = 1e-3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
):
    print(f"Launching training: epochs={epochs}, lr={learning_rate}, bs={batch_size}")
    result = train.remote(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    print(result)
    print("\nTo download checkpoints:")
    print("  modal volume get apple-fm-toolkit /checkpoints/ ./adapter/")
