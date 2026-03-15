"""Modal deployment for Apple FM Thai adapter training.

Setup (one-time):
    pip install modal
    modal setup
    modal volume create apple-fm-toolkit
    modal volume create apple-fm-thai
    modal volume put apple-fm-toolkit /path/to/apple-fm-toolkit/ /toolkit/
    modal volume put apple-fm-thai /path/to/data/processed/iteration_1/ /data/iteration_1/

Run training:
    modal run modal_train.py
    modal run modal_train.py --epochs 1         # quick test
    modal run modal_train.py --iteration 2      # different dataset variant

Download checkpoints:
    modal volume get apple-fm-thai /checkpoints/iteration_1/ ./adapter/iteration_1/

Volumes:
    apple-fm-toolkit  — shared base model + tokenizer (~12GB, reuse across experiments)
    apple-fm-thai     — experiment data + checkpoints (per iteration)
"""

import modal

app = modal.App("apple-fm-thai")

# Shared: base model + tokenizer (reuse across all experiments)
toolkit_volume = modal.Volume.from_name("apple-fm-toolkit")

# Experiment: data + checkpoints (per iteration/variant)
experiment_volume = modal.Volume.from_name("apple-fm-thai")

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

TOOLKIT_MOUNT = "/toolkit"
EXPERIMENT_MOUNT = "/experiment"


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
    iteration: int = 1,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
):
    import subprocess
    import os

    # Locate toolkit (shared) and data (per iteration)
    toolkit_dir = f"{TOOLKIT_MOUNT}/toolkit"
    train_data = f"{EXPERIMENT_MOUNT}/data/iteration_{iteration}/train.jsonl"
    eval_data = f"{EXPERIMENT_MOUNT}/data/iteration_{iteration}/eval.jsonl"
    checkpoint_dir = f"{EXPERIMENT_MOUNT}/checkpoints/iteration_{iteration}"

    # Verify assets exist
    assert os.path.exists(os.path.join(toolkit_dir, "assets", "base-model.pt")), \
        f"Toolkit not found. Upload with: modal volume put apple-fm-toolkit /path/to/apple-fm-toolkit/ /toolkit/"
    assert os.path.exists(train_data), \
        f"Data not found for iteration {iteration}. Upload with: modal volume put apple-fm-thai /path/to/data/ /data/iteration_{iteration}/"

    print(f"=== Iteration {iteration} ===")
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
    print(f"Download: modal volume get apple-fm-thai /checkpoints/iteration_{iteration}/ ./adapter/iteration_{iteration}/")

    return "Training complete"


@app.local_entrypoint()
def main(
    iteration: int = 1,
    epochs: int = 3,
    learning_rate: float = 1e-3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
):
    print(f"Launching iteration {iteration}: epochs={epochs}, lr={learning_rate}, bs={batch_size}")
    result = train.remote(
        iteration=iteration,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    print(result)
    print(f"\nTo download checkpoints:")
    print(f"  modal volume get apple-fm-thai /checkpoints/iteration_{iteration}/ ./adapter/iteration_{iteration}/")
