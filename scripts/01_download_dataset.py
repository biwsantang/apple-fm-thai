"""Download Typhoon datasets from HuggingFace."""

from pathlib import Path

from datasets import load_dataset
from rich.console import Console

console = Console()
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # --- Primary: SFT split (357K rows) ---
    sft_path = RAW_DIR / "typhoon_sft"
    if sft_path.exists():
        console.print(f"[yellow]Skipping SFT download — already exists at {sft_path}[/]")
    else:
        console.print("[bold]Downloading typhoon-s-instruct-post-training (sft split)...[/]")
        ds_sft = load_dataset(
            "typhoon-ai/typhoon-s-instruct-post-training",
            split="sft",
            trust_remote_code=True,
        )
        ds_sft.save_to_disk(str(sft_path))
        console.print(f"[green]Saved {len(ds_sft):,} SFT rows to {sft_path}[/]")

    # --- Secondary: Sovereign dataset (splits have different schemas, load each separately) ---
    # The SFT split uses "messages" format (compatible with our pipeline).
    # NitiBench splits use "prompt" + "reward_model" format (different schema).
    # We only need the SFT-compatible splits for adapter training.
    SOVEREIGN_SPLITS = {
        "nitibench_train_sft": "typhoon_sovereign_nitibench_sft",
        "mirage_test": "typhoon_sovereign_mirage",
    }
    for split_name, save_name in SOVEREIGN_SPLITS.items():
        split_path = RAW_DIR / save_name
        if split_path.exists():
            console.print(f"[yellow]Skipping {split_name} — already exists at {split_path}[/]")
        else:
            console.print(f"[bold]Downloading sovereign split: {split_name}...[/]")
            try:
                ds_split = load_dataset(
                    "typhoon-ai/typhoon-s-sovereign-capability-dataset",
                    split=split_name,
                    trust_remote_code=True,
                )
                ds_split.save_to_disk(str(split_path))
                console.print(f"[green]Saved {len(ds_split):,} rows to {split_path}[/]")
            except Exception as e:
                console.print(f"[red]Failed to load {split_name}: {e}[/]")
                console.print("[yellow]Trying without schema enforcement...[/]")
                from datasets import load_dataset as ld
                ds_split = ld(
                    "typhoon-ai/typhoon-s-sovereign-capability-dataset",
                    data_files=f"*{split_name}*.parquet",
                    split="train",
                    trust_remote_code=True,
                )
                ds_split.save_to_disk(str(split_path))
                console.print(f"[green]Saved {len(ds_split):,} rows to {split_path}[/]")

    console.print("\n[bold green]Download complete.[/]")


if __name__ == "__main__":
    main()
