"""Modal deployment for Apple FM Thai adapter training.

Setup (one-time):
    pip install modal
    modal setup                    # authenticate
    modal volume create apple-fm-toolkit

Upload toolkit assets (one-time, ~12GB):
    modal volume put apple-fm-toolkit /path/to/apple-fm-toolkit/ /toolkit/

Run training:
    modal run modal_train.py
    modal run modal_train.py --epochs 1    # quick test
    modal run modal_train.py --gpu a100    # use A100 instead of H100
"""

import modal

app = modal.App("apple-fm-thai")

# Persistent volume for the 12GB toolkit assets (upload once, reuse forever)
toolkit_volume = modal.Volume.from_name("apple-fm-toolkit")

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.6",
        "tamm~=0.1.0",
        "sentencepiece>=0.2.1",
        "pydantic>=2.12.5",
        "tqdm>=4.67.3",
        "rich>=14.3.3",
        "datasets",
        "huggingface_hub",
        gpu="a100",
    )
    .run_commands(
        "pip install uv",
    )
)

REPO_URL = "https://github.com/biwsantang/apple-fm-thai.git"
TOOLKIT_MOUNT = "/toolkit"
WORK_DIR = "/work"


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={TOOLKIT_MOUNT: toolkit_volume},
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

    # Clone repo
    print(f"=== Cloning {REPO_URL} ===")
    subprocess.run(
        ["git", "clone", REPO_URL, WORK_DIR],
        check=True,
    )

    # Symlink toolkit into the project
    toolkit_src = f"{TOOLKIT_MOUNT}/toolkit"
    toolkit_dst = f"{WORK_DIR}/apple-fm-toolkit"
    # The volume stores under /toolkit/toolkit/ because we uploaded the dir
    # Find the actual assets location
    for candidate in [
        f"{TOOLKIT_MOUNT}/toolkit",
        f"{TOOLKIT_MOUNT}/apple-fm-toolkit",
        TOOLKIT_MOUNT,
    ]:
        if os.path.exists(os.path.join(candidate, "assets", "base-model.pt")):
            toolkit_src = candidate
            break

    print(f"=== Toolkit found at {toolkit_src} ===")
    print(f"    Assets: {os.listdir(os.path.join(toolkit_src, 'assets'))}")
    os.symlink(toolkit_src, toolkit_dst)

    # Install project deps
    print("=== Installing dependencies ===")
    subprocess.run(
        ["uv", "sync", "--no-dev"],
        cwd=WORK_DIR,
        check=True,
    )

    # Download and process data
    print("=== Downloading dataset ===")
    subprocess.run(
        ["uv", "run", "python", "scripts/01_download_dataset.py"],
        cwd=WORK_DIR,
        check=True,
    )

    print("=== Processing dataset ===")
    subprocess.run(
        ["uv", "run", "python", "scripts/03_filter_and_convert.py"],
        cwd=WORK_DIR,
        check=True,
    )

    # Train
    print("=== Starting training ===")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{toolkit_dst}:{env.get('PYTHONPATH', '')}"
    env["PYTHONHASHSEED"] = "42"

    cmd = [
        "uv", "run", "python", "-m", "examples.train_adapter",
        "--train-data", f"{WORK_DIR}/data/processed/iteration_1/train.jsonl",
        "--eval-data", f"{WORK_DIR}/data/processed/iteration_1/eval.jsonl",
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
        "--checkpoint-dir", f"{WORK_DIR}/adapter/iteration_1/",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=WORK_DIR, env=env)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    # List checkpoints
    checkpoint_dir = f"{WORK_DIR}/adapter/iteration_1"
    if os.path.exists(checkpoint_dir):
        files = os.listdir(checkpoint_dir)
        print(f"\n=== Checkpoints ({len(files)} files) ===")
        for f in sorted(files):
            size = os.path.getsize(os.path.join(checkpoint_dir, f))
            print(f"  {f}: {size / 1024 / 1024:.1f} MB")

    # Copy checkpoints to volume for persistence
    output_dir = f"{TOOLKIT_MOUNT}/checkpoints/iteration_1"
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(checkpoint_dir):
        for f in os.listdir(checkpoint_dir):
            src = os.path.join(checkpoint_dir, f)
            dst = os.path.join(output_dir, f)
            shutil.copy2(src, dst)
            print(f"  Saved {f} to volume")

    toolkit_volume.commit()
    print(f"\n=== Done! Checkpoints saved to volume at /checkpoints/iteration_1/ ===")
    print("Download with: modal volume get apple-fm-toolkit /checkpoints/ ./checkpoints/")

    return "Training complete"


@app.local_entrypoint()
def main(
    epochs: int = 3,
    gpu: str = "A100-80GB",
    learning_rate: float = 1e-3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
):
    # Override GPU if requested
    if gpu.lower() != "h100":
        train.gpu = gpu

    print(f"Launching training on {gpu}: epochs={epochs}, lr={learning_rate}, bs={batch_size}")
    result = train.remote(
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    print(result)
    print("\nTo download checkpoints:")
    print("  modal volume get apple-fm-toolkit /checkpoints/ ./adapter/")
